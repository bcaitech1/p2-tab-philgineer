import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
from datetime import date
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score
from gensim.models import Word2Vec


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']


'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label


def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te


# def word2vec_feature(prefix, data_input, groupby, target, size):
#     df = data_input.copy()
    
#     df_bag = pd.DataFrame(df[[groupby, target]])
#     df_bag[target] = df_bag[target].astype(str)
#     df_bag[target].fillna('NAN', inplace=True)
#     df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list':(lambda x: list(x))}).reset_index()
#     doc_list = list(df_bag['list'].values)
#     w2v = Word2Vec(doc_list, window=3, min_count=1, workers=32, size=size)
#     vocab_keys = list(w2v.wv.vocab.keys())
#     w2v_array = []
#     for v in vocab_keys :
#         w2v_array.append(list(w2v.wv[v]))
#     df_w2v = pd.DataFrame()
#     df_w2v['vocab_keys'] = vocab_keys    
#     df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
#     df_w2v.columns = [target] + ['w2v_%s_%s_%d'%(prefix,target,x) for x in range(size)]
#     print ('df_w2v:' + str(df_w2v.shape))
#     return df_w2v


def all_mean_category(df,year_month):
    df = df.copy()
    first_buy = df.drop_duplicates('customer_id', keep='first')[['order_date','customer_id']].sort_values('customer_id').reset_index(drop=True)
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')

    #지금까지 한달에 평균 얼마씩 썼는지 구하기
    #처음 물건 산 뒤로 몇달 지났는지 계산하여 나눠줌
    first_buy['month']=first_buy['order_date'].map(lambda x:12*dateutil.relativedelta.relativedelta(d, x).years+dateutil.relativedelta.relativedelta(d, x).months).reset_index(drop=True)
    data_agg_sum = df.groupby('customer_id').sum().reset_index().sort_values('customer_id')
    first_buy['mean'] = data_agg_sum['total']/(first_buy['month']+1)
    #first_buy['mean'] = first_buy['mean'].map()
    return first_buy['mean']

def count_over(df,year_month):
    cid = 'customer_id'
    cust = df[df['year_month'] < year_month][cid].unique()
    count = pd.DataFrame({'customer_id':cust}).sort_values(cid)
    
    count['count'] = 0
    count = count.set_index(cid)
    while True:
        year_month = date(int(year_month.split('-')[0]),int(year_month.split('-')[1]),1) - dateutil.relativedelta.relativedelta(months=1)
        year_month = year_month.strftime('%Y-%m')
        total = df[df['year_month'] == year_month].groupby('customer_id').sum().reset_index()
        if total.empty: break
        cust = total[total['total']>300][cid].unique()
        for x in cust:
            count.at[x,'count']+=1
    return count.sort_values('customer_id').reset_index()['count']


def feature_engineering1(df, year_month):
    df = df.copy()
    
#     word2vec_fe = word2vec_feature("description", df, "customer_id", "description", 2)
#     df = df.merge(word2vec_fe, on=['description'])

    
    ### new features ###
    df['month'] = df['order_date'].dt.month
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df['order_ts'] = df['order_date'].astype(np.int64) // 1e9
    df['order_ts_diff'] = df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff'] = df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff'] = df.groupby(['customer_id'])['price'].diff()
    df['total_diff'] = df.groupby(['customer_id'])['total'].diff()
    
    df['first_buy'] = all_mean_category(df, year_month)
#     df['over300_month'] = count_over(df, year_month)
    
    
    ### cumsum features 생성, 추가 ###
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()
    
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()
    
    df['cumsum_total_by_year_month'] = df.groupby(['year_month'])['total'].cumsum()
    df['cumsum_quantity_by_year_month'] = df.groupby(['year_month'])['quantity'].cumsum()
    df['cumsum_price_by_year_month'] = df.groupby(['year_month'])['price'].cumsum()
    
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']
    
    mode_func = [lambda x:x.value_counts().index[0]]
    
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'cumsum_total_by_year_month': agg_func,
        'cumsum_quantity_by_year_month': agg_func,
        'cumsum_price_by_year_month': agg_func,
        
        'first_buy': agg_func,
#         'over300_month': agg_func,
        
        # order_id/ product_id 의 nunique
        'order_id' : ['nunique'],
        'product_id' : ['nunique'],

        # 월/년월 최빈도 + 기본 agg
        'month' : mode_func + agg_func,  # ['mean', 'max', ..., + mode_func]
        'year_month' : mode_func,

        # Time-Series diff 
        'order_ts' : ['first', 'last'],
        'order_ts_diff' : agg_func, # 구매 내역 사이의 시간 차
        'quantity_diff' : agg_func,
        'price_diff' : agg_func,
        'total_diff' : agg_func
    }
    
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features


if __name__ == '__main__':
    
    print('data_dir', data_dir)
