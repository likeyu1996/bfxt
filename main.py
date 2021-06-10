#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 李珂宇
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa import stattools
from statsmodels.tsa import arima_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_te
from scipy import stats

# numpy完整print输出
np.set_printoptions(threshold=np.inf)
# pandas完整print输出
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

is_raw = pd.read_csv(filepath_or_buffer='Data/IS_600111.CSV', encoding='gb18030', na_values=np.nan)
# is_raw.rename(columns={str(is_raw.columns[0]): "Name"})
cache_columns = list(is_raw.columns)
cache_columns[0] = "Name"
is_raw.columns = cache_columns
# print(is_raw)

translator_dict = {
    "Total_Revenue": "一、营业总收入(百万元)",
    "Operating_Revenue": "营业收入(百万元)",
    "Total_Cost": "二、营业总成本(百万元)",
    "Operating_Cost": "营业成本(百万元)",
    "R&D_Cost": "研发费用(百万元)",
    "Tax_Cost": "税金及附加(百万元)",
    "Sale_Cost": "销售费用(百万元)",
    "Manage_Cost": "管理费用(百万元)",
    "Finance_Cost": "财务费用(百万元)",
    "Impairment_Loss": "资产减值损失(百万元)",
    "OOI": "三、其他经营收益",
    "OOI_1": "加:公允价值变动收益(百万元)",
    "OOI_2": "加:投资收益(百万元)",
    "OOI_3": "资产处置收益(百万元)",
    "OOI_4": "资产减值损失(新)(百万元)",
    "OOI_5": "信用减值损失(新)(百万元)",
    "OOI_6": "其他收益(百万元)",
    "OOI_7": "营业利润平衡项目(百万元)",
    "Operating_Profit": "四、营业利润(百万元)",
    "Non_Operating_Revenue": "加:营业外收入(百万元)",
    "Non_Operating_Cost": "减:营业外支出(百万元)",
    "Profit_Balance": "利润总额平衡项目(百万元)",
    "Total_Profit": "五、利润总额(百万元)",
    "Tax": "减:所得税费用(百万元)",
    "Net_Profit": "六、净利润(百万元)",
    "Net_Operating_Profit": "扣除非经常性损益后的净利润(百万元)",
    "OCI": "八、其他综合收益(百万元)",
    "Accumulated_Profit": "九、综合收益总额(百万元)",
}
inverse_translator_dict = {value: key for key, value in translator_dict.items()}
# for i in range(is_raw["Name"].size):
# print(is_raw["Name"].str.contains("年报"))
df_0 = is_raw.loc[is_raw.loc[:, "Name"].str.contains("年报")]
df_0.loc[:, "Name"] = [i[:4] for i in df_0.loc[:, "Name"].to_numpy()]
df_0.index = [i for i in range(df_0.loc[:, "Name"].size)]
# print(df_0)
df_1 = is_raw.loc[is_raw.loc[:, "Name"].str.contains("中报")]
df_1.loc[:, "Name"] = [i[:4] for i in df_1.loc[:, "Name"].to_numpy()]
df_1.index = [i for i in range(df_1.loc[:, "Name"].size)]
# print(df_1)
year_list = [i for i in range(2019, 2009, -1)]
# print(year_list)
# 计算半年一年倍数
multiple_dict = {}
for i in is_raw.columns:
    # print([df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0] for j in year_list if not df_0.loc[df_0.loc[:, "Name"] == str(j), i].empty and not isinstance(df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0], str)])
    # print([df_1.loc[df_1.loc[:, "Name"] == str(j), i].to_numpy()[0] for j in year_list if not df_1.loc[df_0.loc[:, "Name"] == str(j), i].empty and not isinstance(df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0], str)])
    # print([type(df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0]) for j in year_list if not df_0.loc[df_0.loc[:, "Name"] == str(j), i].empty])
    # print([isinstance(df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0], str) for j in year_list if not df_0.loc[df_0.loc[:, "Name"] == str(j), i].empty])
    # print(type(df_0.loc[df_0.loc[:, "Name"] == str(j), i]) for j in year_list)
    '''
    list_cache = []
    for j in year_list:
        atom_0_cache = df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0]
        atom_1_cache = df_1.loc[df_1.loc[:, "Name"] == str(j), i].to_numpy()[0]
        # print('atom_0_cache ', atom_0_cache)
        # print('atom_1_cache ', atom_1_cache)
        if df_0.loc[df_0.loc[:, "Name"] == str(j), i].empty or \
                df_1.loc[df_1.loc[:, "Name"] == str(j), i].empty or \
                isinstance(atom_0_cache, str) or isinstance(atom_1_cache, str) or \
                atom_0_cache == np.nan or atom_1_cache == np.nan or atom_1_cache == 0.0:
            data_cache = np.nan
        else:
            data_cache = atom_0_cache / atom_1_cache
        list_cache.append(data_cache)
    multiple_dict[i] = list_cache
    '''
    multiple_dict[i] = [np.nan
                        if df_0.loc[df_0.loc[:, "Name"] == str(j), i].empty or
                           df_1.loc[df_1.loc[:, "Name"] == str(j), i].empty or
                           isinstance(df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0], str) or
                           isinstance(df_1.loc[df_1.loc[:, "Name"] == str(j), i].to_numpy()[0], str) or
                           df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0] == np.nan or
                           df_1.loc[df_1.loc[:, "Name"] == str(j), i].to_numpy()[0] == np.nan or
                           df_1.loc[df_1.loc[:, "Name"] == str(j), i].to_numpy()[0] == 0.0
                        else (df_0.loc[df_0.loc[:, "Name"] == str(j), i].to_numpy()[0] /
                        df_1.loc[df_1.loc[:, "Name"] == str(j), i].to_numpy()[0])
                        for j in year_list]
multiple_dict['Name'] = year_list
df_double = pd.DataFrame(pd.DataFrame(multiple_dict).mean()).T
# print(df_double)
# print([df_double.loc[:,i].to_numpy()[0] for i in is_raw.columns])
data_2020 = [is_raw.loc[is_raw.loc[:, "Name"].str.contains("2020"), i].to_numpy()[0] *
             df_double.loc[:, i].to_numpy()[0]
             if isinstance(is_raw.loc[is_raw.loc[:, "Name"].str.contains("2020"), i].to_numpy()[0],np.float) and
                isinstance(df_double.loc[:, i].to_numpy()[0], np.float)
             else np.nan for i in is_raw.columns]
# print(dict(zip(is_raw.columns,data_2020)))


def dict_clean(dic,year):
    dic['Name'] = str(year)
    dic['上市前/上市后'] = '上市后'
    dic['报表类型'] = '合并报表'
    dic['公司类型'] = '通用'
    dic['公告日期'] = '0'
    dic['数据来源'] = '年度报告'
    dic['审计意见(境内)'] = '标准无保留意见'
    return dic


dict_2020 = dict_clean(dict(zip(is_raw.columns,data_2020)),2020)
df_2020 = pd.DataFrame(dict_2020,index=[0])
# If using all scalar values, you must pass an index
df_2 = pd.concat([df_2020, df_0], axis=0, ignore_index=True)
# df_0.loc[df_0.shape[0]] = dict(zip(df_0.columns,data_2020))
# print(df_2)
df_2.to_csv('Result/IS_Year.CSV', encoding='gb18030')
# 得到2020年报预测数据
# 计算增长率
growth_dict = {}
for i in is_raw.columns:
    growth_dict[i] = [np.nan if isinstance(df_2.loc[j, i], str) or isinstance(df_2.loc[j+1, i], str) or
                           df_2.loc[j, i] == np.nan or df_2.loc[j+1, i] == np.nan or df_2.loc[j+1, i] == 0.0
                      else (df_2.loc[j, i] / df_2.loc[j+1, i] - 1.0)
                      for j in df_2.index[:-1]]
growth_dict['Name'] = df_2.loc[:df_2.index.size - 2, 'Name'].to_numpy()
# print(df_2.loc[:df_2.index.size - 2, 'Name'].to_numpy())
df_growth = pd.DataFrame(growth_dict)
# print(df_growth)
df_growth.to_csv('Result/IS_Growth.CSV', encoding='gb18030')
# 增长率计算完毕
def growth_chart():
    # 绘制增长率折线图
    for i in range(df_growth.columns.size):
        if df_growth.columns[i] in inverse_translator_dict:

            # data_cache = df_growth.loc[:, ['Name', df_growth.columns[i]]].set_index('Name').sort_values(by='Name', ascending=True)
            # cache_columns = list(data_cache.columns)
            # cache_columns[0] = inverse_translator_dict[df_growth.columns[i]]
            # data_cache.columns = cache_columns
            # print(data_cache)

            # df_growth_inverse = df_growth.set_index('Name').sort_values(by='Name', ascending=True)
            # plt.figure(figsize=[16, 9])
            df_growth_cache = df_growth.copy(deep=True)
            df_growth.fillna(0.0,inplace=True)
            sns.set(rc={'figure.figsize':(16,9)})
            sns.lineplot(x='Name', y=df_growth_cache.columns[i], data=df_growth_cache, marker='*')
            plt.xlabel('Year')
            plt.ylabel('Growth')
            plt.title(inverse_translator_dict[df_growth_cache.columns[i]])
            bonus_cache = np.mean(df_growth_cache[df_growth_cache.columns[i]].to_numpy())/10.0
            for x, y in zip(df_growth_cache['Name'].to_numpy(), df_growth_cache[df_growth_cache.columns[i]].to_numpy()):
                # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
                plt.text(x=x,  # x-coordinate position of data label
                         y=y+bonus_cache,  # y-coordinate position of data label, adjusted to be 150 below the data point
                         s='{:.2f}'.format(y),  # data label, formatted to ignore decimals
                         color='purple')  # set colour of line
            plt.savefig('./Chart/' + str(i) + '_' + df_growth_cache.columns[i].replace(':', '_') + '.png')
            plt.close('all')
            # 增长率图绘制完毕
# growth_chart()
# TODO:困了 以后改成通用形式
def profit(g_r,g_c):
    p_0 = df_2.iloc[0,4]-df_2.iloc[0,6]
    p_1 = df_2.iloc[0,4]*(1+g_r[0])-df_2.iloc[0,6]*(1+g_c[0])
    p_2 = df_2.iloc[0,4]*(1+g_r[0])*(1+g_r[1])-df_2.iloc[0,6]*(1+g_c[0])*(1+g_c[1])
    p_3 = df_2.iloc[0,4]*(1+g_r[0])*(1+g_r[1])*(1+g_r[2])-df_2.iloc[0,6]*(1+g_c[0])*(1+g_c[1])*(1+g_c[2])
    g_1 = p_1/p_0-1
    g_2 = p_2/p_1-1
    g_3 = p_3/p_2-1
    return [g_1,g_2,g_3]
def w_mean_predict(past_num,predict_num,df,col):
    growth_gross_list = [df.loc[i, df.columns[col]] for i in range(past_num)]
    for j in range(predict_num):
        growth_gross = 0
        for k in range(past_num):
            growth_gross += (past_num-k) * growth_gross_list[k]
        growth_gross *= 2.0/(past_num**2+past_num)
        growth_gross_list.insert(0, growth_gross)
    return growth_gross_list[:predict_num][::-1]
'''
g_op_r_w= w_mean_predict(5,3,df_growth,4)
g_op_c_w= w_mean_predict(5,3,df_growth,6)
print(profit(g_op_r_w,g_op_c_w))
'''


def tsa_predict(predict_num,df,col):
    growth_gross_list = df.loc[:, df.columns[col]].to_numpy()[::-1]
    '''
    ic_max_ar = 7
    ic_max_ma = 3
    growth_aic = stattools.arma_order_select_ic(growth_gross_list,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='aic')
    growth_bic = stattools.arma_order_select_ic(growth_gross_list,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='bic')
    print(growth_aic,growth_bic)
    '''
    growth_model = arima_model.ARMA(growth_gross_list,(2,1)).fit()
    # print(growth_model.summary())
    return growth_model.predict(start=len(growth_gross_list), end=len(growth_gross_list)+predict_num-1, dynamic=True)
'''
g_op_r_t = tsa_predict(3,df_growth,4)
g_op_c_t = tsa_predict(3,df_growth,6)
print(profit(g_op_r_t,g_op_c_t))
'''

