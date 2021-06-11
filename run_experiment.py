import torch
from tqdm.auto import tqdm
import random
from IPython.display import set_matplotlib_formats
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
import numpy
import os
import pickle
import fire
import itertools
import copy
from scipy import stats

class LSTM_model(torch.nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            config['true_hs'], config['true_hs'], config['true_nl'], batch_first=True, dropout=0.0
        )
        self.projection = torch.nn.Linear(config['true_hs'], config['vocab_size']+1)  # +1 PAD

        self.lt = torch.nn.Embedding(config['vocab_size']+2, config['true_hs'])  # +2 PAD and BOS (we need BOS to be able to score the first token)

        # set pad token bias to negative inf
        with torch.no_grad():
            self.projection.bias[config['pad_token_id']] = -float("Inf")

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def forward(self, x, hidden=None):
        x = self.lt(x)
        lstm_out, hidden_out = self.lstm(x, hidden)
        output = self.projection(lstm_out)

        return output, hidden_out


@torch.no_grad()
def score_space_tensor(
    space_tensor, model, config,
):
    device = config['device']
    scoring_batch_size = config['scoring_batch_size']
    bos_token_id = config['bos_token_id']
    pad_token_id = config['pad_token_id']

    space_log_scores = []
    model = model.to(device)

    space_dataset = TensorDataset(space_tensor)
    space_loader = DataLoader(space_dataset, batch_size=scoring_batch_size, shuffle=False)

    for minibatch in tqdm(space_loader):
        minibatch = minibatch[0]
        mbsize = minibatch.size(0)
        bos_prefix = torch.tensor([[bos_token_id]] * mbsize)
        bossed_tensor = torch.cat([bos_prefix, minibatch], dim=1)
        inp = bossed_tensor[:, :-1]  # cut the last token from the input
        target = bossed_tensor[:, 1:]  # cut the first token from target        

        output, _ = model(inp.to(device))
        logp = torch.log_softmax(output, dim=-1)

        target_logp = (
            torch.gather(logp, dim=-1, index=target.unsqueeze(-1).to(logp.device))
            .squeeze(-1)
            .cpu()
        )
        pad_mask = (target == pad_token_id)
        target_logp[pad_mask] = 0.0  # change -inf in pads to 0. which will not affect the sumlog

        space_log_scores.append(target_logp)

    space_log_scores = torch.cat(space_log_scores, dim=0)

    return space_log_scores

def create_training_data(ground_truth_logprobs, sequence_space, config):
    """
    computes empirical distribution and training/valid data
    fow now train/valid split may overlap
    """

    num_valid_samples = int(config["training_set_size"] * 0.05)

    ground_truth_logprobs_temperature = ground_truth_logprobs / config["empirical_temperature"]
    split_dict = {}

    for split_name, num_samples in zip(['train', 'valid'], [config["training_set_size"], num_valid_samples]):
        empirical_sampled_ids = ground_truth_logprobs_temperature.softmax(dim=0).multinomial(num_samples, replacement=True)
        split_dict[split_name] = empirical_sampled_ids

    combined_ids = torch.cat([split_dict['train'], split_dict['valid']], dim=0)
    empirical_unique_ids, empirical_unique_counts = combined_ids.unique(dim=0, sorted=True, return_counts=True)

    training_tensor = torch.index_select(sequence_space, dim=0, index=split_dict['train'])
    validation_tensor = torch.index_select(sequence_space, dim=0, index=split_dict['valid'])

    return empirical_unique_ids, empirical_unique_counts, split_dict['train'], training_tensor, split_dict['valid'], validation_tensor


def train_model(training_model, train_tensor, valid_tensor, config):
    bos_token_id = config['bos_token_id']
    pad_token_id = config['pad_token_id']
    device = config['device']
    vocab_size = config['vocab_size']
    train_batch_size = config['train_batch_size']
    train_max_epochs = config['train_max_epochs']
    train_max_steps = config['train_max_steps']
    valid_every_nsteps = config['valid_every_nsteps']
    lr = config['lr']
    lr_scheduler_patience = config['lr_scheduler_patience']
    lr_scheduler_factor = config['lr_scheduler_factor']
    early_stop_patience = config['early_stop_patience']

    bos_prefix = torch.tensor([bos_token_id], dtype=torch.int)
    train_tensor = torch.cat([bos_prefix.repeat(train_tensor.size(0), 1), train_tensor], dim=1)
    valid_tensor = torch.cat([bos_prefix.repeat(valid_tensor.size(0), 1), valid_tensor], dim=1)

    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    optimizer = Adam(training_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=lr_scheduler_patience, factor=lr_scheduler_factor)
    loss = CrossEntropyLoss(ignore_index=pad_token_id)
    training_model = training_model.to(device)
    early_stop_patience_counter = 0
    lowest_val_loss = 1e3
    scheduler.step(lowest_val_loss)  # to create ._last_lr
    epoch_number = 0
    early_stop = False
    reach_max_steps = False
    acc_moving_avg = 0
    trloss_moving_avg = 0
    val_acc = 0
    best_epoch = -1
    best_epoch_step = -1
    checkpoints_path = os.path.join(config['save_dir'],f'best_model_{config["grid_step"]}.pt')
    for epoch_number in range(300):
        bar = tqdm(train_loader)
        train_loss_log = []
        for batch_step, batch in enumerate(bar):
            total_step = epoch_number*len(bar) + batch_step
            if total_step > train_max_steps:
                reach_max_steps = True
                break
            bar.set_description_str(f'Early stop {early_stop_patience_counter}/{early_stop_patience}, total step: {total_step} Tr mov loss: {trloss_moving_avg:.7f} tr mov acc: {acc_moving_avg:.7f} val acc: {val_acc:.7f} current best val loss {lowest_val_loss:.7f}, lr {scheduler._last_lr}')
            inp = batch[0][:, :-1].to(device)  # cut last time step since we don't have label for T+1
            target = batch[0][:, 1:].to(device) # cut the first time step since there is no predicted score for it
            output, _ = training_model(inp)
            _, max_tokens = output.max(dim=-1)
            pad_mask = target != pad_token_id
            num_nonpad_tokens = pad_mask.sum().item()
            correct_nonpad = (max_tokens == target) * pad_mask
            acc = (correct_nonpad.sum().float() / num_nonpad_tokens).item()
            acc_moving_avg = acc_moving_avg * 0.1 + acc * 0.9
            celoss = loss(output.view(-1, vocab_size+1), target.reshape(-1))  # +2 to address EOS, PAD in the proj layer
            train_loss_log.append(celoss.item())
            trloss_moving_avg = trloss_moving_avg * 0.1 + celoss.item() * 0.9
            celoss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_step % valid_every_nsteps == 0 or batch_step == (len(bar)-1):
                val_loss, val_acc = validation_step(valid_tensor, training_model, config)
                scheduler.step(val_loss)
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_epoch = epoch_number
                    best_epoch_step = batch_step
                    torch.save(training_model.state_dict(), checkpoints_path)
                    early_stop_patience_counter = 0
                else:
                    early_stop_patience_counter += 1
                if early_stop_patience_counter > early_stop_patience:
                    early_stop = True
                    if early_stop:
                        break
        if early_stop:
            break
        if reach_max_steps:
            break

    training_model.load_state_dict(torch.load(checkpoints_path))
    val_loss, val_acc = validation_step(valid_tensor, training_model, config)
    assert val_loss == lowest_val_loss

    if config['remove_model']:
        os.remove(checkpoints_path)

    train_stat = {
        'best_val_loss': lowest_val_loss,
        'best_val_acc': val_acc,
        'best_epoch': best_epoch,
        'best_epoch_step': best_epoch_step,
        'epochs_processed': epoch_number,
        'early_stop': early_stop,
        'max_step': reach_max_steps,
    }
                        
    return training_model, train_stat

@torch.no_grad()
def validation_step(valid_tensor, training_model, config):
    bos_token_id = config['bos_token_id']
    pad_token_id = config['pad_token_id']
    device = config['device']
    vocab_size = config['vocab_size']
    valid_batch_size = config['valid_batch_size']

    training_model.eval()
    valid_dataset = TensorDataset(valid_tensor)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=valid_batch_size)
    loss = CrossEntropyLoss(ignore_index=pad_token_id)
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_tokens = 0
    for batch in tqdm(valid_loader, desc='Validation'):
        inp = batch[0][:, :-1].to(device)  # cut last time step since we don't have label for T+1
        target = batch[0][:, 1:].to(device) # cut the first time step since there is no predicted score for it
        output, _ = training_model(inp)
        _, max_tokens = output.max(dim=-1)
        pad_mask = target != pad_token_id
        num_nonpad_tokens = pad_mask.sum().item()
        correct_nonpad = (max_tokens == target) * pad_mask
        total_correct += correct_nonpad.sum().item()
        total_tokens += num_nonpad_tokens
        celoss = loss(output.view(-1, vocab_size+1), target.reshape(-1))
        total_loss += celoss.item()
        total_batches += 1

    total_loss = total_loss / total_batches
    total_acc = total_correct / total_tokens
    training_model.train()

    return total_loss, total_acc

def create_model(config):
    training_model = LSTM_model(config)
    maybe_saved_model_path = os.path.join(config['save_dir'],f'best_model_{config["grid_step"]}.pt')
    loaded = False
    if os.path.exists(maybe_saved_model_path) and config["try_load_model"]:
        training_model.load_state_dict(torch.load(maybe_saved_model_path))
        training_model.to(config['device'])
        training_model.eval()
        loaded = True

    return training_model, loaded

def top_p_filtering(scores_tensor, top_p):
    sorted_logits, sorted_indices = torch.sort(scores_tensor, descending=True)
    cumulative_probs = torch.cumsum(sorted_logits.exp(), dim=-1)

    # Remove tokens with cumulative top_p above the threshold 
    sorted_indices_to_keep = cumulative_probs <= top_p
    seq_id_to_keep = sorted_indices[sorted_indices_to_keep]

    return seq_id_to_keep.tolist()

def counts_top_p_filtering(sorted_counts, sorted_seq_ids, top_p):
    """
    counts assumed to be sorted with descending=True
    """
    cumulative_probs = sorted_counts.cumsum(dim=0) / sorted_counts.sum().float()
    sorted_indices_to_keep = cumulative_probs <= top_p
    seq_id_to_keep = sorted_seq_ids[sorted_indices_to_keep]

    return seq_id_to_keep.tolist()


def kl(left_d_logp, right_d_logp):
    log_ratio = left_d_logp - right_d_logp
    kl = (left_d_logp.exp() * log_ratio).sum()
    return kl

def generate_sequence_space(config):
    # empty sequence first
    vocab_size = config['vocab_size']
    eos_token_id = config['eos_token_id']
    pad_token_id = config['pad_token_id']
    max_length = config['max_length']
    voc = torch.tensor(list(range(vocab_size-1)))
    
    length_spaces = []
    for iter_length in range(1, max_length):
        cartesian_argument = [voc] * iter_length
        sequence_space = torch.cartesian_prod(*cartesian_argument)
        if sequence_space.ndim == 1:
            sequence_space = sequence_space.unsqueeze(1)
        suffix = torch.tensor([eos_token_id]+[pad_token_id]*((max_length-1)-iter_length)).unsqueeze(0).repeat(sequence_space.shape[0], 1)
        length_spaces.append(torch.cat([sequence_space, suffix], dim=1))
    
    empty_seq = torch.tensor([eos_token_id]+[pad_token_id]*(max_length-1))
    space = torch.cat([empty_seq.unsqueeze(0)]+length_spaces, dim=0)
    config['space_num_seqs'] = space.size(0)

    return space

def compute_cost_linear(p_array, q_array, k, maximum=-1):
    p_topk = p_array[:k]
    if q_array.size < p_topk.size:
        return maximum
    for kprime in range(p_topk.size, q_array.size):
        if p_topk.size == numpy.intersect1d(p_topk, q_array[:kprime]).size:
            return kprime
    return maximum

def compute_cost_binarysearch_topp(p_array_seqids, p_array_logprobs, q_array_seqids, q_array_logprobs, eps, maximum=-1):
    k_mapped = ((numpy.exp(p_array_logprobs).cumsum() > eps) == 0).sum() + 1
    k_prime = compute_cost_binarysearch(p_array_seqids, q_array_seqids, k_mapped, maximum=maximum)
    if k_prime == maximum:
        eps_prime = 1.
    else:
        eps_prime = numpy.exp(q_array_logprobs)[:k_prime].sum()

    return eps_prime, k_mapped, k_prime

def compute_cost_binarysearch(p_array, q_array, k, maximum=-1, lower_bound=None):
    p_topk = p_array[:k]
    actual_k = p_topk.size
    left_bound = p_topk.size if lower_bound is None else lower_bound
    right_bound = q_array.size
    if right_bound < actual_k:
        return maximum

    # naive check if there is perfect match
    actual_k_intersection_size = numpy.intersect1d(p_topk, q_array[:actual_k]).size
    if actual_k_intersection_size == actual_k:
        # kprime is k
        return actual_k
    
    # check if given lower bound works here too
    if lower_bound:
        lower_bound_intersection_size = numpy.intersect1d(p_topk, q_array[:lower_bound]).size
        if lower_bound_intersection_size == actual_k:
            return lower_bound

    elif q_array.size == actual_k:
        # it is not the perfect match but |q| == k, so it equals to |Sigma^{\to L}|
        return maximum
    done = False
    while not done:
        kprime = left_bound + (right_bound - left_bound) // 2
        intersection_size = numpy.intersect1d(p_topk, q_array[:kprime]).size
        if intersection_size < actual_k:
            # move kprime to the right
            left_bound = kprime
        elif intersection_size == actual_k:
            # need to check if kprime is minimized
            intersection_size_decremented = numpy.intersect1d(p_topk, q_array[:(kprime-1)]).size
            if intersection_size_decremented < actual_k:
                # kprime is minimized
                done = True
            else:
                # move kprime to the left
                right_bound = kprime
        if right_bound - left_bound == 1:
            # bounds clashed, cost is full support then
            done = True
            kprime = maximum
            
        # for debug purposes
        #print(f'kprime={kprime} ({left_bound}, {right_bound})')
        #import ipdb; ipdb.set_trace()
    return kprime

def compute_cost_binarysearch_new(p_array, q_array, k, maximum=-1, lower_bound=None):
    p_topk = p_array[:k]
    actual_k = p_topk.size
    left_bound = actual_k if lower_bound is None or actual_k > lower_bound else lower_bound
    right_bound = q_array.size

    # first check the actual left_bound (lower bound may give a match already)
    left_bound_intersection_size = numpy.intersect1d(p_topk, q_array[:left_bound]).size
    if left_bound_intersection_size == actual_k:
        return left_bound

    # if |supp(q)| <= left_bound then max, or equal since if above would be true if there is a match
    if right_bound <= left_bound:
        return maximum


    finished = False
    lowest_kprime = maximum
    
    while not finished:
        kprime = left_bound + ((right_bound - left_bound) // 2)
        intersection_size = numpy.intersect1d(p_topk, q_array[:kprime]).size

        if intersection_size < actual_k:
            # increase left bound, go right
            left_bound = kprime
        elif intersection_size == actual_k:
            # decrease left bound, go left
            right_bound = kprime
            lowest_kprime = kprime

        #print(f'kprime={kprime} ({left_bound}, {right_bound}) \cap = {intersection_size}')
        if right_bound - left_bound == 1:
            # this intersection size is the final one
            finished = True
        
    return lowest_kprime
    

def compute_cost_binarysearch_multik(p_array, q_array, ks, maximum=-1, tqdm_disabled=True):
    costs = []
    lower_bound = None
    for k in tqdm(ks, disable=tqdm_disabled):
        cost = compute_cost_binarysearch_new(p_array, q_array, k, maximum=maximum, lower_bound=lower_bound)
        costs.append(cost)
        lower_bound = cost
    
    return numpy.array(costs)

def compute_intersection_union_multik(p_array, q_array, ks, tqdm_disabled=True):
    intersections = []
    unions = []
    for k in tqdm(ks, disable=tqdm_disabled):
        p_topk = p_array[:k]
        support_stopk_intersection = numpy.intersect1d(p_topk, q_array).size
        support_stopk_union = numpy.union1d(p_topk, q_array).size
        intersections.append(support_stopk_intersection)
        unions.append(support_stopk_union)

    return numpy.array(intersections), numpy.array(unions)

    
def compute_cost_df(row, pname, qname, k, solver='bs'):
    # to use in notebooks and pandas apply function
    p_array = numpy.array(row[pname])
    q_array = numpy.array(row[qname])

    if solver == 'bs':
        cost = compute_cost_binarysearch(p_array, q_array, k)
    elif solver == 'linear':
        cost = compute_cost_linear(p_array, q_array, k)
    
    return cost


def get_seq_id(seq: list, config):
    """
    expected format : [..., EOS]
    """

    seq = seq[:-1]
    seq_length = len(seq)
    if seq_length == 0:
        return 0

    position = 0
    for base, token in enumerate(reversed(seq)):
        position += token * ((config['vocab_size']-1)**base)  # config['vocab_size']-1 since EOS is not in cart product
        if token == config['pad_token_id']:
            raise Exception('PAD is not allowed to appear in the sequence!')

    for length in range(0, seq_length):
        position += (config['vocab_size']-1)**length

    if position >= config['space_num_seqs']:
        raise Exception('position can not go beyond seq space!')

    return position

@torch.no_grad()
def beam_search_decoding(model, beam_size, config):
    """
    not the fastest version of beam search: hidden activations are not cached between time steps
    not that we do not have infinite seq handling at the moment (i.e. for loop finished without EOS being chosen), it never happened so far
    """

    beam_size_start = beam_size
    finished_sequences = []
    finished_scores = [] 

    partial_tokens = torch.tensor([config['bos_token_id']], dtype=torch.long, device=config['device'])[None,:]  # no beam size expanding needed for the first time step
    partial_seq_logprob = torch.tensor([0.], device=config['device'])

    for t in range(config['max_length']):
        output, _ = model(partial_tokens)

        output_last_ts = output[:,-1,:]
        next_logprob = output_last_ts.log_softmax(dim=-1)  # [beam_size, vocab_size]
        next_logprob = next_logprob[:,:-1]  # exclude pad token, assumed to be the last one
        #next_logprob[:,config['pad_token_id']] = -1e5
        extended_logprobs = partial_seq_logprob.unsqueeze(-1) + next_logprob

        extended_topk_logprobs, extended_argtopk = torch.topk(extended_logprobs.view(-1), k=min(beam_size, extended_logprobs.view(-1).numel()), dim=0)
        argtop_token_id = extended_argtopk % next_logprob.shape[-1]
        argtop_hypothesis_id = extended_argtopk // next_logprob.shape[-1]

        # permute partial arrays according to the argtop_hypothesis_id
        non_eos_tails = (argtop_token_id != config['eos_token_id']).nonzero(as_tuple=True)[0]
        

        reordered_partial_tokens = torch.index_select(partial_tokens, dim=0, index=argtop_hypothesis_id)
        partial_tokens_all = torch.cat([reordered_partial_tokens, argtop_token_id.unsqueeze(1)], dim=1)  # this one includes EOS hyps
        partial_tokens = partial_tokens_all[non_eos_tails]
        partial_seq_logprob = extended_topk_logprobs[non_eos_tails]

        eos_tails = (argtop_token_id == config['eos_token_id']).nonzero(as_tuple=True)[0]
        eos_partial_tokens = partial_tokens_all[eos_tails]
        seq_with_eos_scores = extended_topk_logprobs[eos_tails]

        finished_sequences.extend(eos_partial_tokens.tolist())
        finished_scores.extend(seq_with_eos_scores)

        beam_size = beam_size - eos_tails.numel()  # decrease beam size up to num of finished hypotheses
        
        if beam_size == 0:
            # assert len(finished_sequences) == beam_size_start  # this may happen if all hyps are EOS but beam size > vocab size
            break

    exact_scores = []
    space_positions = []

    # computing the score the scores, here with pads in output layer
    for seq, score in zip(finished_sequences, finished_scores):
        seq_tensor = torch.tensor([seq], device=config['device'])
        inp = seq_tensor[:, :-1]
        target = seq_tensor[:, 1:]
        output, _ = model(inp)
        logprob = output.log_softmax(dim=-1)  # [1, time, vocab]
        
        seq_logprob = torch.gather(logprob, dim=-1, index=target.unsqueeze(-1)).sum()
        assert torch.allclose(seq_logprob, score)
        exact_scores.append(seq_logprob)
        space_positions.append(get_seq_id(seq[1:], config))  # seq[1:] to trim BOS

    exact_scores = torch.stack(exact_scores,dim=0)
    # renormalize scores to get induced distr
    induced_distr_scores = exact_scores - exact_scores.logsumexp(dim=0)

    return finished_sequences, space_positions, induced_distr_scores

@torch.no_grad()
def ancestral_sampling_decoding(model, anc_num_samples, anc_temperature, config):
    """
    samlping a token step by step
    sampled seqs are not scored here, instead we return correspodning space seq_ids such that we retrieve seq_logprobs from whole space model scores later
    """

    seq_ids = []

    max_num_iterations = 5000
    num_iteration = 0
    num_nonterminated = 0
    

    while len(set(seq_ids)) < anc_num_samples:
        num_left_to_sample = anc_num_samples - len(set(seq_ids))
        bos = torch.tensor([config['bos_token_id']], dtype=torch.long, device=config['device'])[None,:].repeat(num_left_to_sample, 1)
        inp = bos
        sampled = [bos.cpu()]
        hidden = None

        for t in range(config['max_length']):
            output, hidden = model(inp, hidden=hidden)
            next_token_probs = (output[:, -1, :]/anc_temperature).softmax(dim=-1)
            inp = next_token_probs.multinomial(1)
            sampled.append(inp.cpu())
            
        sampled = torch.cat(sampled, dim=1)
        
        # find first EOS occurrence
        tmp_arange = torch.arange(sampled.size(1), 0, -1)
        weigthed_eos_positions = (sampled == config['eos_token_id']) * tmp_arange
        first_eos_position = weigthed_eos_positions.argmax(dim=1) * (weigthed_eos_positions.sum(1) > 0)  # product ensures 0 is no EOS in the seq    

        # trimming after EOS
        eos_trimed = []
        
        for seq, eos_pos in zip(sampled, first_eos_position):
            if eos_pos == 0:
                # sequence was not terminated, skipping it
                # seq = torch.cat([seq, torch.tensor([config['eos_token_id']])], dim=0).to(config['device'])
                num_nonterminated += 1
                continue
            else:
                seq = seq[:eos_pos+1].to(config['device'])  # +1 to get the EOS in

            eos_trimed.append(seq)
            sampled_seq_id = get_seq_id(seq.tolist()[1:], config)  # seq.tolist()[1:] 1: to trim BOS
            seq_ids.append(sampled_seq_id)

        num_iteration += 1
        if num_iteration == max_num_iterations:
            break
        
    seq_ids = torch.tensor(seq_ids)
    seq_ids_unique, seq_ids_counts = seq_ids.unique(return_counts=True)

    return seq_ids_unique, seq_ids_counts, num_iteration


def laplace_score_space_tensor(sequence_space, config):
    """
    sample each seq prob independently from laplace
    """
    num_sequences = sequence_space.size(0)
    laplace_d = torch.distributions.laplace.Laplace(torch.tensor([config['laplace_loc']]), torch.tensor([config['laplace_scale']]))
    log_probs = laplace_d.sample([num_sequences]).view(-1).log_softmax(dim=0)

    return log_probs

def create_gt(sequence_space: torch.FloatTensor, config: dict) -> torch.FloatTensor:
    random_lstm = LSTM_model(config)
    random_lstm.eval()  # always eval mode

    lstm_scored_tokens_lprobs = score_space_tensor(sequence_space, random_lstm, config)
    lstm_scored_seq_lprobs = lstm_scored_tokens_lprobs.sum(1) - lstm_scored_tokens_lprobs.sum(1).logsumexp(dim=0)


    if config['gt_second_distr'] == 'lstm_same':
        second_distr_seq_lprobs = lstm_scored_tokens_lprobs.mean(1) - lstm_scored_tokens_lprobs.mean(1).logsumexp(dim=0)
    elif config['gt_second_distr'] == 'lstm_new':
        random_lstm_2 = LSTM_model(config)
        random_lstm_2.eval()  # always eval mode
        lstm_scored_tokens_lprobs_2 = score_space_tensor(sequence_space, random_lstm_2, config)
        second_distr_seq_lprobs = lstm_scored_tokens_lprobs_2.mean(1) - lstm_scored_tokens_lprobs_2.mean(1).logsumexp(dim=0)
    elif config['gt_second_distr'] == 'laplace':
        second_distr_seq_lprobs = laplace_score_space_tensor(sequence_space, config)

    if config['gt_interpolation_mode'] == 'p':
        interpolated_scores = config['gt_interpolation'] * lstm_scored_seq_lprobs.exp() + (1 - config['gt_interpolation']) * second_distr_seq_lprobs.exp()
        interpolated_scores = interpolated_scores.log()
    elif config['gt_interpolation_mode'] == 'logp':
        interpolated_scores = config['gt_interpolation'] * lstm_scored_seq_lprobs + (1 - config['gt_interpolation']) * second_distr_seq_lprobs

    return interpolated_scores


def run_experiment(**config):
    # setting up sequence space
    torch.manual_seed(config["seed"])
    numpy.random.seed(config["seed"])
    random.seed(config["seed"])
    # torch.set_deterministic(True)  # turn on if you want synced cuda for debugging
    results = {}

    sequence_space = generate_sequence_space(config)

    print(f"Space size: {sequence_space.size(0)} seqs")

    ground_truth_logprobs = create_gt(sequence_space, config)
    ground_truth_logprobs = ground_truth_logprobs - ground_truth_logprobs.logsumexp(dim=0)
    results["true_distr_entropy"] = -(ground_truth_logprobs * ground_truth_logprobs.exp()).sum().item()
    
    sampled_seq_ids, sampled_counts, train_seq_ids, train_tensor, valid_seq_ids, valid_tensor = create_training_data(ground_truth_logprobs, sequence_space, config)
    emp_sorted_counts, emp_count_positions = sampled_counts.sort(descending=True)
    emp_topk_seq_ids = sampled_seq_ids[emp_count_positions]  # permute sampled ids w.r.t. sorted ordering

    train_length_stats = stats.describe((train_tensor != config['pad_token_id']).sum(dim=1))
    results["train_length_minmax"] = train_length_stats.minmax
    results["train_length_mean"] = train_length_stats.mean
    results["train_length_var"] = train_length_stats.variance

    if config['topk_modes_to_save'] > 0:
        true_topk_seq_logprobs, true_topk_seq_ids = torch.topk(ground_truth_logprobs, k=int(config['topk_modes_to_save']), largest=True)
    else:
        true_topk_seq_logprobs, true_topk_seq_ids = ground_truth_logprobs.sort(descending=True)
    
    results[f"top_seq_ids_true"] = true_topk_seq_ids.numpy()
    results[f"top_seq_lprobs_true"] = true_topk_seq_logprobs.numpy()
    results[f"top_seq_ids_emp"] = emp_topk_seq_ids.numpy()
    results[f"top_seq_counts_emp"] = emp_sorted_counts.numpy()
    
    if config['only_empirical'] == False:
        # training the model or loading the saved one
        model, loaded_from = create_model(config)
        if not loaded_from:
            model, train_stat = train_model(model, train_tensor, valid_tensor, config)
        else:
            val_loss, val_acc = validation_step(valid_tensor, model, config)
            train_stat = {}
            print(f'Loaded model from {loaded_from} \n val loss = {val_loss:.7f}')

        model.eval()  # turning off potential dropout
        
        model_token_space_logprobs = score_space_tensor(sequence_space, model, config)
        model_sequence_space_logprobs = model_token_space_logprobs.sum(dim=1)

        if config['topk_modes_to_save'] > 0:
            model_topk_seq_logprobs, model_topk_seq_ids = torch.topk(model_sequence_space_logprobs, k=int(config['topk_modes_to_save']), largest=True)
        else:
            model_topk_seq_logprobs, model_topk_seq_ids = model_sequence_space_logprobs.sort(descending=True)
        results[f"top_seq_ids_model"] = model_topk_seq_ids.numpy()
        results[f"top_seq_lprobs_model"] = model_topk_seq_logprobs.numpy()

        

        for bsize in config['beam_size']:
            beam_sequences, beam_seq_ids, beam_seq_induced_logprobs = beam_search_decoding(model, bsize, config)
            beam_seq_induced_logprobs_sorted, beam_seq_induced_logprobs_argsorted = beam_seq_induced_logprobs.sort(descending=True)
            beam_seq_ids_argsorted = torch.index_select(torch.tensor(beam_seq_ids), dim=0, index=beam_seq_induced_logprobs_argsorted.cpu())
            results[f"support_size_beam_{bsize}"] = len(beam_seq_ids_argsorted)
            results[f"top_seq_ids_beam_{bsize}"] = beam_seq_ids_argsorted.numpy()
            results[f"top_seq_lprobs_beam_{bsize}"] = beam_seq_induced_logprobs_sorted.cpu().numpy()

        for anc_num_samples in config['ancestral_num_samples']:
            for anc_temperature in config['ancestral_temperature']:
                # ancestral sampling, extra sorting w.r.t. log probs is additionaly performed
                anc_t_str = str(anc_temperature).replace('.','')
                sampling_seq_ids, sampling_seq_counts, num_iteration = ancestral_sampling_decoding(model, anc_num_samples, anc_temperature, config)  # sampling_seq_ids sorted w.r.t. counts (not actual model logprobs)!
                sampling_seq_logprobs = torch.index_select(model_sequence_space_logprobs, dim=0, index=sampling_seq_ids)
                sampling_seq_logprobs = sampling_seq_logprobs - sampling_seq_logprobs.logsumexp(dim=0)  # renomralizing in log space
                sampling_seq_logprobs_sorted, sampling_seq_logprobs_sort_ids = sampling_seq_logprobs.sort(descending=True)
                sampling_seq_ids_logprobsorted = sampling_seq_ids[sampling_seq_logprobs_sort_ids].numpy()
                results[f"anc_{anc_num_samples}_t_{anc_t_str}_num_iterations"] = num_iteration
                results[f"support_size_anc_{anc_num_samples}_t_{anc_t_str}"] = len(sampling_seq_ids_logprobsorted)
                results[f"top_seq_ids_anc_{anc_num_samples}_t_{anc_t_str}"] = sampling_seq_ids_logprobsorted
                results[f"top_seq_lprobs_anc_{anc_num_samples}_t_{anc_t_str}"] = sampling_seq_logprobs_sorted.numpy()
                results[f"top_seq_counts_anc_{anc_num_samples}_t_{anc_t_str}"] = sampling_seq_counts[sampling_seq_logprobs_sort_ids].numpy()

        divergences = {}
        kl_true_model = kl(ground_truth_logprobs, model_sequence_space_logprobs)
        results['KL(true, model)'] = kl_true_model.item()
        results['model_distr_entropy'] = -(model_sequence_space_logprobs * model_sequence_space_logprobs.exp()).sum().item()

        for k,v in train_stat.items():
            results[k] = v

    results['num_train_uniq_seqs'] = train_seq_ids.unique().numel()
    results['num_valid_uniq_seqs'] = valid_seq_ids.unique().numel()
    results['support_size_emp'] = emp_topk_seq_ids.numel()
    

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    results['config'] = config
    save_path = os.path.join(config['save_dir'], config['save_name'])
    pickle.dump(results, open(save_path, 'wb'))
    print(f'saved to {save_path}')
    
def config_factory(grid_step=0):

    gt_confs = {
        'laplace_only': {
            "gt_interpolation": 0.0,
            "gt_interpolation_mode": 'p',
        },
        'lstm_only': {
            "gt_interpolation": 1.0,
            "gt_interpolation_mode": 'p',
        },
        'alpha_03_mode_logp': {
            "gt_interpolation": 0.3,
            "gt_interpolation_mode": 'logp',
        },
    }



    grid = {
        'seed': [7820, 5578, 6978, 6195, 4809, 6317, 8651, 1907, 5822, 8925],
        "true_config": list(gt_confs.keys()),  
        "model_hs": [128, 512],
        "training_set_size": [int(1e5), int(5*1e5), int(1e6), int(5*1e6), int(1e7)],
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    
    step_grid = grid_setups[grid_step - 1]  # slurm var starts from 1

    print(f"Running {step_grid} config out of {len(grid_setups)}...")

    if torch.cuda.device_count() > 0:
        expr_device = "cuda"
    else:
        expr_device = "cpu"

    selected_true_config = gt_confs[step_grid['true_config']]
    # selected_model_config = model_confs[step_grid['model_config']]

    # space specs
    vocab_size = 7
    max_length = 10


    config = {
        "grid_step": grid_step,
        "device": expr_device,
        "seed": step_grid["seed"],
        "only_empirical": False,
        # space config
        "vocab_size": vocab_size,  # includes EOS
        "max_length": max_length,  # includes EOS
        "eos_token_id": vocab_size - 1,
        "pad_token_id": vocab_size,
        "bos_token_id": vocab_size + 1,
        "scoring_batch_size": int(1e3),
        # ground truth distribution
        "true_hs": 512, #step_grid["true_hs"],
        "true_nl": 2,
        "gt_interpolation": selected_true_config["gt_interpolation"],
        "gt_interpolation_mode": selected_true_config["gt_interpolation_mode"],
        "gt_second_distr": 'laplace', #step_grid["gt_second_distr"],
        "laplace_loc": 0.0,
        "laplace_scale": 1.0,
        # empirical distribution
        "empirical_temperature": 1.0,
        "empirical_topk": -1,
        "training_set_size": step_grid["training_set_size"],
        # learned distribution
        "model_hs": step_grid["model_hs"],
        "model_nl": 2,
        "model_dropout": 0.1,
        "try_load_model": False,
        "remove_model": True,
        # training loop
        'train_batch_size': 1024*5,
        'train_max_steps': int(2e4),
        'train_max_epochs': 300,
        'valid_every_nsteps': 500,
        'early_stop_patience': 5,
        'lr': 0.0001,
        'lr_scheduler_patience': 1,
        'lr_scheduler_factor': 0.8,
        # validation loop
        'valid_batch_size': 1024,
        # decoding strategies
        'beam_size': [100, 150, 200, 250, 300, 350, 400, 450, 500],
        'ancestral_num_samples': [100, 150, 200, 250, 300, 350, 400, 450, 500],
        'ancestral_temperature': [1.0],
        # save
        "topk_modes_to_save": -1,
        "save_dir": "PUT_SAVE_PATH_HERE",
        "save_name": f"metric_step_{grid_step}.pkl",
    }

    return config


def main(grid_step):
    config = config_factory(grid_step)
    print(f"step config\n {config}")
    run_experiment(**config)

if __name__ == "__main__":
    fire.Fire(main)