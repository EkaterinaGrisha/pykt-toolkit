# _*_ coding:utf-8 _*_

import pandas as pd
from .utils import sta_infos, write_txt, format_list2str

KEYS = ["user_id", "skill_id", "problem_id"]

def read_data_from_csv(read_file, write_file):
    stares = []

    # kt-research patch P3: the canonical skill_builder_data_corrected_collapsed.csv
    # is ISO-8859-1 (latin-1) encoded (non-ASCII bytes in skill_name); reading it as
    # utf-8 raises UnicodeDecodeError. latin-1 is the documented encoding for this file.
    df = pd.read_csv(read_file, encoding='ISO-8859-1', dtype=str)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df['tmp_index'] = range(len(df))
    _df = df.dropna(subset=["user_id","problem_id", "skill_id", "correct", "order_id"])

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    # kt-research patch P4: pandas >=2 returns tuple group keys for groupby([col]),
    # so str(user) became "('64525',)" and corrupted the written sequence file
    # (downstream int(seq_len) then failed). Grouping by the scalar column name
    # restores scalar keys, matching pyKT's original (old-pandas) assumption.
    ui_df = _df.groupby("user_id", sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=['order_id','tmp_index'])
        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problem_id'].tolist()
        seq_skills = tmp_inter['skill_id'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_start_time = ['NA']
        seq_response_cost = ['NA']

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), format_list2str(seq_skills), format_list2str(seq_ans), seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

