import math
import pandas as pd
import pymssql as pms
import geopy.distance
from itertools import combinations, permutations


def combination_create(city, int_choose, dis_km):  # 城市 ‘XX市’，选几个, 公里数换算
    dis = dis_km / 100
    server = "10.110.3.56"
    user = "sa"
    password = "1qaz2wsx3EDC"
    database = "city_test"
    conn = pms.connect(server, user, password, database)

    strSQL = "SELECT * FROM [city_test].[planning].[candidate_list] where [市] = '" + city + "'"
    df_ca = pd.read_sql(strSQL, conn)
    df_ca['id'] = df_ca['id'].astype(int)
    # 生成组合列表
    int_base = len(df_ca)
    i = 0
    a = [i for i in range(1, int_base + 1)]
    df_com = pd.DataFrame(combinations(a, int_choose))
    # df_com = df_com.sort_index(ascending=False)
    df_com['t'] = 0
    df_com['c'] = 0
    df_pick = pd.DataFrame(combinations(a, 2))
    df_pick['s'] = df_pick.index
    df_pick['d'] = 0

    i = 0
    for i in range(len(df_pick)):
        df_combination = df_ca[df_ca['id'].isin(df_pick.iloc[i, 0:2])]
        if abs(df_combination.iloc[0, 3] - df_combination.iloc[1, 3]) <= dis:
            df_pick['d'][df_pick['s'] == i] = 1
            continue
        if abs(df_combination.iloc[0, 2] - df_combination.iloc[1, 2]) <= dis:
            df_pick['d'][df_pick['s'] == i] = 1
            continue
        if (abs(df_combination.iloc[0, 2] - df_combination.iloc[1, 2]) + abs(
                df_combination.iloc[0, 3] - df_combination.iloc[1, 3])) < (2 ** 0.5 * dis):
            df_pick['d'][df_pick['s'] == i] = 1
            continue
        if geopy.distance.distance((df_combination.iloc[0, 3], df_combination.iloc[0, 2]),
                                   (df_combination.iloc[1, 3], df_combination.iloc[1, 2])).km <= dis_km:
            df_pick['d'][df_pick['s'] == i] = 1
            continue

    # 取出不在一定范围内的值
    df_val = df_pick[df_pick['d'] == 1]
    i = 0
    c = [i for i in range(int_choose)]
    df_matrix = pd.DataFrame(combinations(c, 2))
    i = 0
    for i in range(len(df_val)):
        j = 0
        for j in range(len(df_matrix)):
            # if df_com.loc[i, 't'] == 0:
            df_com['t'][(df_com.loc[:, df_matrix.iloc[j, 0]] == df_val.iloc[i, 0]) & (
                    df_com.loc[:, df_matrix.iloc[j, 1]] == df_val.iloc[i, 1])] = 1

    df_result = df_com[df_com['t'] == 0]

    df_co = pd.DataFrame(df_result.iloc[0, 0:int_choose])
    df_co.columns = ['company_id']
    df_co['group'] = 1

    for i in range(1, len(df_result) - 1):
        temp = pd.DataFrame(df_result.iloc[i, 0:int_choose])
        temp.columns = ['company_id']
        temp['group'] = i + 1
        df_co = df_co.append(temp)

    return df_co


df_input = combination_create('深圳市', 4, 5)