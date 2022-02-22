import tqdm 
import json
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
import pickle as pkl
import argparse


def save_obj(name, obj):
    print('saving to name...')
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def read_user_rating_records():
    print('reading raw data file...')
    if 'Amazon' in args.dataset:
        col_names = ['user_id', 'item_id', 'rating', 'timestamp']
        data_records = pd.read_csv(dir_path + rating_file, sep=',', names=col_names, engine='python')
    elif args.dataset == 'yelp2020':
        line_count = len(open(dir_path + rating_file).readlines())
        print(line_count)
        user_ids, business_ids, stars, dates = [], [], [], []
        with open(dir_path + rating_file) as f:
            for line in tqdm.tqdm(f, total=line_count):       
                blob = json.loads(line)
                user_ids += [blob['user_id']]
                business_ids += [blob['business_id']]
                stars += [blob['stars']]
                dates += [blob['date']]
        data_records = pd.DataFrame({'user_id': user_ids, 'item_id': business_ids, 'rating': stars, 'timestamp': dates})

        sorted_data_records = data_records.sort_values(by=['timestamp'])
        sorted_data_records = sorted_data_records.iloc[-4000000:]
        data_records = data_records[data_records.index.isin(sorted_data_records.index)]

    return data_records


def remove_infrequent_items(data, min_counts=5):
    df = deepcopy(data)
    counts = df['item_id'].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_counts].index)]
    print("items with < {} interactoins are removed".format(min_counts))
    return df


def remove_infrequent_users(data, min_counts=10):
    df = deepcopy(data)
    counts = df['user_id'].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_counts].index)]
    print("users with < {} interactoins are removed".format(min_counts))
    return df


def generate_inverse_mapping(data_list):
    inverse_mapping = dict()
    for inner_id, true_id in enumerate(data_list):
        inverse_mapping[true_id] = inner_id
    return inverse_mapping


def convert_to_inner_index(user_records, user_mapping, item_mapping):
    inner_user_records = []
    user_inverse_mapping = generate_inverse_mapping(user_mapping)
    item_inverse_mapping = generate_inverse_mapping(item_mapping)

    for user_id in range(len(user_mapping)):
        real_user_id = user_mapping[user_id]
        item_list = list(user_records[real_user_id])
        for index, real_item_id in enumerate(item_list):
            item_list[index] = item_inverse_mapping[real_item_id]
        inner_user_records.append(item_list)
    return inner_user_records, user_inverse_mapping, item_inverse_mapping


def generate_rating_matrix(train_set, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, article_list in enumerate(train_set):
        for article in article_list:
            row.append(user_id)
            col.append(article)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
    return rating_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['Amazon-CD', 'Amazon-Book', 'Amazon-Beauty', 'Amazon-Movies', 'yelp2020', 'Amazon-Digital', 'Amazon-Clothing', 'Amazon-Kindle', 'Amazon-Sports'], default='Amazon-Sports')
    parser.add_argument("--read_path", type=str, default='./amazon-sports/')
    args = parser.parse_args()

    dir_path = args.read_path
    if args.dataset == 'Amazon-CD':
        rating_file = 'ratings_CDs_and_Vinyl.csv'
        params = [10, 5]
    elif args.dataset == 'Amazon-Beauty':
        rating_file = 'ratings_Beauty.csv'
        params = [10, 5]
    elif args.dataset == 'Amazon-Movies':
        rating_file = 'ratings_Movies_and_TV.csv'
        params = [10, 5]
    elif args.dataset == 'Amazon-Sports':
        rating_file = 'ratings_Sports_and_Outdoors.csv'
        params = [10, 5]
    elif args.dataset == 'yelp2020':
        rating_file = 'yelp_academic_dataset_review.json'
        params = [10, 5]
    else:
        raise NotImplementedError("dataset %s is not included." % args.dataset)

    data_records = read_user_rating_records()

    # data_records.to_csv(dir_path + args.dataset + '_rating.csv', sep='\t', encoding='utf-8')
    
    # data_records.loc[data_records.rating < 4, 'rating'] = 0
    # data_records.loc[data_records.rating >= 4, 'rating'] = 1
    # data_records = data_records[data_records.rating > 0]

    print("start preprocessing %s dataset..." % args.dataset)
    print('filtering users...')
    filtered_data = remove_infrequent_users(data_records, params[0])
    print('filtering items...')
    filtered_data = remove_infrequent_items(filtered_data, params[0])
    print('filtering users...')
    filtered_data = remove_infrequent_users(filtered_data, params[1])

    data = filtered_data.groupby('user_id')['item_id'].apply(list)
    unique_data = filtered_data.groupby('user_id')['item_id'].nunique()
    data = data[unique_data[unique_data >= params[1]].index]

    csv_dir = dir_path + 'filtered_rating.csv'
    filtered_data.reset_index(drop=True)
    filtered_data.to_csv(csv_dir, sep='\t', columns=['user_id', 'item_id', 'rating', 'timestamp'], header=False, index=False)

    # a user may rate the same item at different timestamps
    # data = filtered_data.groupby('user_id')['item_id'].apply(list)
    # unique_data = filtered_data.groupby('user_id')['item_id'].nunique()
    # data = data[unique_data[unique_data >= params[1]].index]

    # user_item_dict = data.to_dict()
    # user_mapping = []
    # item_set = set()
    # for user_id, item_list in data.iteritems():
    #     user_mapping.append(user_id)
    #     for item_id in item_list:
    #         item_set.add(item_id)
    # item_mapping = list(item_set)

    # print('num of users:{}, num of items:{}'.format(len(user_mapping), len(item_mapping)))
    # inner_data_records, user_inverse_mapping, item_inverse_mapping = convert_to_inner_index(user_item_dict,
    #                                                                                         user_mapping, item_mapping)

    # rating_matrix = generate_rating_matrix(inner_data_records, len(user_mapping), len(item_mapping))
    # rating_matrix = generate_rating_matrix(inner_data_records, user_mapping, item_mapping)
    # # rating_matrix = rating_matrix.transpose()
    # print(rating_matrix.nnz)
    
    # if not os.path.exists('../data/' + args.dataset):
    #     os.makedirs('../data/' + args.dataset)
    
    # save_obj('./' + args.dataset + '/user_item_list', inner_data_records)

    # save_obj('./' + args.dataset + 'rating', rating_matrix)
