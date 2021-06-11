import pandas as pd
import numpy
import pickle
from glob import glob
from tqdm.auto import tqdm
import math
import torch
from run_experiment import (
    compute_cost_binarysearch_topp,
    compute_cost_binarysearch_multik,
    compute_cost_binarysearch,
    compute_intersection_union_multik,
)
import fire

import multiprocessing
from joblib import Parallel, delayed
import os


def parse_pkl_multik(fname, tqdm_disabled=True):
    results = pickle.load(open(fname, "rb"))
    save = {}

    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, dict):
            continue
        if isinstance(v, list):
            continue
        if isinstance(v, numpy.ndarray):
            continue
        save[k] = v

    anc_temperatures = [
        str(i).replace(".", "") for i in results["config"]["ancestral_temperature"]
    ]
    decoding_widths = results["config"]["beam_size"]
    space_num_seqs = results["config"]["space_num_seqs"]

    k_list = [1, 2, 3, 4] + list(range(5, 505, 5))
    save["k_list"] = k_list

    top_seq_lprobs_emp = (
        torch.tensor(
            results["top_seq_counts_emp"] / results["top_seq_counts_emp"].sum()
        )
        .log()
        .numpy()
    )
    save["top_seq_ids_emp"] = results["top_seq_ids_emp"]
    save["top_seq_lprobs_emp"] = top_seq_lprobs_emp
    save["top_seq_ids_true_10k"] = results["top_seq_ids_true"][:10000]
    save["top_seq_lprobs_true_10k"] = results["top_seq_lprobs_true"][:10000]

    save[f"cost_true_emp"] = compute_cost_binarysearch_multik(
        results["top_seq_ids_true"],
        results["top_seq_ids_emp"],
        k_list,
        maximum=space_num_seqs,
        tqdm_disabled=tqdm_disabled,
    )
    save[f"cost_true_model"] = compute_cost_binarysearch_multik(
        results["top_seq_ids_true"],
        results["top_seq_ids_model"],
        k_list,
        maximum=space_num_seqs,
        tqdm_disabled=tqdm_disabled,
    )
    save[f"cost_emp_model"] = compute_cost_binarysearch_multik(
        results["top_seq_ids_emp"],
        results["top_seq_ids_model"],
        k_list,
        maximum=space_num_seqs,
        tqdm_disabled=tqdm_disabled,
    )

    for dec_width in decoding_widths:
        save[f"cost_true_beam{dec_width}"] = compute_cost_binarysearch_multik(
            results["top_seq_ids_true"],
            results[f"top_seq_ids_beam_{dec_width}"],
            k_list,
            maximum=space_num_seqs,
            tqdm_disabled=tqdm_disabled,
        )
        save[f"cost_model_beam{dec_width}"] = compute_cost_binarysearch_multik(
            results["top_seq_ids_model"],
            results[f"top_seq_ids_beam_{dec_width}"],
            k_list,
            maximum=space_num_seqs,
            tqdm_disabled=tqdm_disabled,
        )
        save[f"cost_emp_beam{dec_width}"] = compute_cost_binarysearch_multik(
            results["top_seq_ids_emp"],
            results[f"top_seq_ids_beam_{dec_width}"],
            k_list,
            maximum=space_num_seqs,
            tqdm_disabled=tqdm_disabled,
        )

        (
            save[f"intersection_true_beam{dec_width}"],
            save[f"union_true_beam{dec_width}"],
        ) = compute_intersection_union_multik(
            results["top_seq_ids_true"],
            results[f"top_seq_ids_beam_{dec_width}"],
            k_list,
            tqdm_disabled=tqdm_disabled,
        )
        (
            save[f"intersection_model_beam{dec_width}"],
            save[f"union_model_beam{dec_width}"],
        ) = compute_intersection_union_multik(
            results["top_seq_ids_model"],
            results[f"top_seq_ids_beam_{dec_width}"],
            k_list,
            tqdm_disabled=tqdm_disabled,
        )
        (
            save[f"intersection_emp_beam{dec_width}"],
            save[f"union_emp_beam{dec_width}"],
        ) = compute_intersection_union_multik(
            results["top_seq_ids_emp"],
            results[f"top_seq_ids_beam_{dec_width}"],
            k_list,
            tqdm_disabled=tqdm_disabled,
        )

        for anc_temp in anc_temperatures:
            save[
                f"cost_true_anc{dec_width}_t_{anc_temp}"
            ] = compute_cost_binarysearch_multik(
                results["top_seq_ids_true"],
                results[f"top_seq_ids_anc_{dec_width}_t_{anc_temp}"],
                k_list,
                maximum=space_num_seqs,
                tqdm_disabled=tqdm_disabled,
            )
            save[
                f"cost_model_anc{dec_width}_t_{anc_temp}"
            ] = compute_cost_binarysearch_multik(
                results["top_seq_ids_model"],
                results[f"top_seq_ids_anc_{dec_width}_t_{anc_temp}"],
                k_list,
                maximum=space_num_seqs,
                tqdm_disabled=tqdm_disabled,
            )
            save[
                f"cost_emp_anc{dec_width}_t_{anc_temp}"
            ] = compute_cost_binarysearch_multik(
                results["top_seq_ids_emp"],
                results[f"top_seq_ids_anc_{dec_width}_t_{anc_temp}"],
                k_list,
                maximum=space_num_seqs,
                tqdm_disabled=tqdm_disabled,
            )

            (
                save[f"intersection_true_anc{dec_width}_t_{anc_temp}"],
                save[f"union_true_anc{dec_width}_t_{anc_temp}"],
            ) = compute_intersection_union_multik(
                results["top_seq_ids_true"],
                results[f"top_seq_ids_anc_{dec_width}_t_{anc_temp}"],
                k_list,
                tqdm_disabled=tqdm_disabled,
            )
            (
                save[f"intersection_model_anc{dec_width}_t_{anc_temp}"],
                save[f"union_model_anc{dec_width}_t_{anc_temp}"],
            ) = compute_intersection_union_multik(
                results["top_seq_ids_model"],
                results[f"top_seq_ids_anc_{dec_width}_t_{anc_temp}"],
                k_list,
                tqdm_disabled=tqdm_disabled,
            )
            (
                save[f"intersection_emp_anc{dec_width}_t_{anc_temp}"],
                save[f"union_emp_anc{dec_width}_t_{anc_temp}"],
            ) = compute_intersection_union_multik(
                results["top_seq_ids_emp"],
                results[f"top_seq_ids_anc_{dec_width}_t_{anc_temp}"],
                k_list,
                tqdm_disabled=tqdm_disabled,
            )

    for k, v in results["config"].items():
        if isinstance(v, dict):
            continue
        save[k] = v

    return save


def not_main(pkl_folder, save_to, num_proc=40, multik=True, tqdm_disabled=True):
    pickle_filenames = glob(f"{pkl_folder}/*pkl")
    print(f"Found {len(pickle_filenames)} in {pkl_folder}.")

    if os.path.exists(save_to):
        print("Save destination exists, exiting...")
        return 1

    pairs = []
    for fname1 in pickle_filenames:
        for fname2 in pickle_filenames:
            if fname1 == fname2:
                continue
            pairs.append((fname1, fname2))

    print(f"Found {len(pairs)} gt pairs.")

    pkls = Parallel(n_jobs=num_proc, verbose=10)(
        delayed(compare_ground_truth)(fname1, fname2, tqdm_disabled=tqdm_disabled)
        for (fname1, fname2) in pairs[:1000]
    )

    df = pd.DataFrame(pkls)
    pickle.dump(df, open(save_to, "wb"))
    print(f"Saved to {save_to}")

    return 0


def main(pkl_folder, save_to, num_proc=40, multik=True, tqdm_disabled=True):
    pickle_filenames = glob(f"{pkl_folder}/*pkl")
    print(f"Found {len(pickle_filenames)} in {pkl_folder}.")

    if os.path.exists(save_to):
        print("Save destination exists, exiting...")
        return 1

    func = parse_pkl_multik
    print(f"Using {func} now...")

    pkls = Parallel(n_jobs=num_proc, verbose=1)(
        delayed(func)(fname, tqdm_disabled=tqdm_disabled) for fname in pickle_filenames
    )

    df = pd.DataFrame(pkls)
    pickle.dump(df, open(save_to, "wb"))
    print(f"Saved to {save_to}")

    return 0


if __name__ == "__main__":
    fire.Fire(main)
