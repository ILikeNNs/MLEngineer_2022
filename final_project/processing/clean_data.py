import pandas as pd
import numpy as np
import math
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import sqlite3
from itertools import combinations
from itertools import chain
from pathlib import Path

def clean_portfolio(p_df):
    """
    cleans the portfolio dataset
    
    INPUT:
        p_df: DataFrame object containing the portfolio data
    
    OUTPUT:
        p_df: DataFrame object containing the cleaned portfolio data
    
    """
    mlb = MultiLabelBinarizer()
    lb = LabelEncoder()

    # one hot encode (for the benchmark model)
    p_df = p_df.join(pd.DataFrame(mlb.fit_transform(p_df['channels']),
                                  index=p_df.index,
                                  columns=mlb.classes_))

    p_df = p_df.join(pd.get_dummies(p_df['offer_type']))
    
    # label encode (for LightGBM's internal categorical handling)
    p_df['channels_string'] = [','.join(map(str, l)) for l in p_df['channels']]
    p_df['channels_int'] = lb.fit_transform(p_df['channels_string'])
    p_df['offer_type_int'] = lb.fit_transform(p_df['offer_type'])
    p_df = p_df.drop('channels_string', axis=1)
    
    # change the duration from days to hours
    p_df['duration_h'] = p_df['duration'] * 24

    return p_df


def clean_profile(p_df):
    """
    cleans the profile dataset
    
    INPUT:
        p_df: DataFrame object containing the profile data
    
    OUTPUT:
        p_df: DataFrame object containing the cleaned profile data
        
    """
    p_df = p_df.dropna()
    p_df = p_df[p_df['age'] != 118].reset_index(drop=True)
    p_df['gender'] = np.where(p_df['gender'] == 'M', 1, 0)
    p_df['year'] = p_df['became_member_on'].astype(str).str[:4]
    return p_df


def clean_transcript(t_df):
    """
    cleans the transcript dataset
    
    INPUT:
        t_df: DataFrame object containing the transcript data
    
    OUTPUT:
        t_df: DataFrame object containing the cleaned transcript data
        
    """
    t_df = pd.concat([t_df, t_df['value'].apply(pd.Series)], axis=1)
    t_df['offer_id'].fillna('', inplace=True)
    t_df['offer id'].fillna('', inplace=True)
    t_df['offer_id'] = t_df['offer_id'] + t_df['offer id']
    t_df = t_df.drop(['offer id', 'value'], axis=1)
    return t_df
    
    
def generate_offer_r_df(t_df, p_df):
    """
    creates the received offer dataset
    
    INPUT:
        t_df: DataFrame object containing the transcript data
        p_df: DataFrame object containing the portfolio data
    
    OUTPUT:
        offer_df_r: DataFrame object containing the received offer data
        
    """
    offer_df_r = t_df[(t_df['event'] == 'offer received')]
    offer_df_r = offer_df_r.drop(['amount'], axis=1)
    offer_df_r = offer_df_r.merge(p_df[['id','duration_h']],
                             how='left', right_on='id', left_on='offer_id')
    offer_df_r = offer_df_r.drop(['id', 'reward'], axis=1)
    offer_df_r['time_ended'] = np.where(offer_df_r['time'] + offer_df_r['duration_h'] - 1 > 714, 
                                        714,offer_df_r['time'] + offer_df_r['duration_h'] - 1)
    return offer_df_r


def generate_offer_v_df(t_df):
    """
    creates the viewed offer dataset
    
    INPUT:
        t_df: DataFrame object containing the transcript data
    
    OUTPUT:
        offer_df_v: DataFrame object containing the viewed offer data
        
    """
    offer_df_v = t_df[(t_df['event'] == 'offer viewed')]
    offer_df_v = offer_df_v.drop(['amount', 'reward'], axis=1).reset_index(drop=True)
    return offer_df_v


def generate_offer_c_df(t_df):
    """
    creates the completed offer dataset
    
    INPUT:
        t_df: DataFrame object containing the transcript data
    
    OUTPUT:
        offer_df_c: DataFrame object containing the completed offer data
        
    """
    offer_df_c = t_df[(t_df['event'] == 'offer completed')]
    offer_df_c = offer_df_c.drop(['amount', 'reward'], axis=1).reset_index(drop=True)
    return offer_df_c


def perform_sql_join(r_df, c_df, v_df, query):
    """
    creates the combined (received+viewed+completed) offer dataset by using sql logic
    
    INPUT:
        r_df: DataFrame object containing the received offer data
        c_df: DataFrame object containing the completed offer data
        v_df: DataFrame object containing the viewed offer data
        query: SQL query to join three dataframes based on time inequality
    
    OUTPUT:
        joined_df: DataFrame object containing the combined offer data
        
    """
    #Make the db in memory
    conn = sqlite3.connect(':memory:')
    
    #write the tables
    r_df.to_sql('received', conn, index=False)
    v_df.to_sql('viewed', conn, index=False)
    c_df.to_sql('completed', conn, index=False)
    joined_df = pd.read_sql_query(query, conn)
    return joined_df


def filter_out_rows(df):
    """
    filters out rows that do not make sense from a logical perspective 
    (offer viewed/completed outside of the validity timeframe)
    
    INPUT:
        df: DataFrame object containing the combined offer data
    
    OUTPUT:
        offer_df: DataFrame object containing the filtered combined offer data
        
    """
    df['min_time_viewed'] = df.groupby(['person', 'offer_id', 'time'])['time_viewed'].transform('min')
    df['min_time_completed'] = df.groupby(['person', 'offer_id', 'time'])['time_completed'].transform('min')
    
    condition1 = (df['min_time_viewed'] == df['time_viewed'])
    condition2 = (df['min_time_viewed'].isnull() == True)
    condition3 = (df['min_time_completed'] == df['time_completed'])
    condition4 = (df['min_time_completed'].isnull() == True)
    

    offer_df = df[(condition1 & condition3) |
                  (condition1 & condition4) |
                  (condition2 & condition3) |
                  (condition2 & condition4)].reset_index(drop=True)
    
    return offer_df


def process_offer_df(df):
    """
    transform the combined offer data by:
    - adding event viewed flag
    - adding event completed flag
    - handling informative campaigns' time completed and event completed flag
    - adding time processed column
    
    
    INPUT:
        df: DataFrame object containing the filtered combined offer data
    
    OUTPUT:
        df: DataFrame object containing the processed offer data
        
    """
    
    # adjust the time viewed and completed
    df['time_viewed'] = np.where((df['time_viewed'] <= df['time_ended']),
                                 df['time_viewed'], np.nan)
    
    df['time_completed'] = np.where((df['time_completed'] <= df['time_ended']),
                                 df['time_completed'], np.nan)
    
    # if time_viewed after end time, set viewed to 0 
    df['event_viewed'] = np.where((df['time_viewed'].isnull() == True) |
                                  (df['time_viewed'] > df['time_ended']) |
                                  (df['time_viewed'] > df['time_completed']),0,1)
    
    # same for completed
    df['event_completed'] = np.where((df['time_completed'].isnull() == True) |
                                  (df['time_completed'] > df['time_ended']),0,1)
    
    # adjust informative offers - they are completed on viewing
    df['event_completed'] = np.where((df['offer_id'].isin(['3f207df678b143eea3cee63160fa8bed', 
                                                           '5a8bc65990b245e5a138643cd4eb9837'])) &
                                      (df['event_viewed'] == 1), 1, df['event_completed'])

    df['time_completed'] = np.where((df['offer_id'].isin(['3f207df678b143eea3cee63160fa8bed', 
                                                       '5a8bc65990b245e5a138643cd4eb9837'])) &
                                      (df['event_viewed'] == 1), df['time_ended'], df['time_completed'])
    
    df['time_processed'] = np.where(df['time_completed'].isnull() == False, df['time_completed'], df['time_ended'])
    df = df.sort_values(by=['person', 'time']).reset_index(drop=True)
    return df


def get_transaction_df(t_df):
    """
    creates the transactional dataset
    
    INPUT:
        t_df: DataFrame object containing the transcript data
        
    """
    return t_df[t_df['event'] == 'transaction'][['person', 'time', 'amount']].reset_index(drop=True)


def get_ranges(offer_df, customer):
    """
    for a given customer, get all timeranges when an offer was active
    
    INPUT:
        offer_df: DataFrame object containing the processed offer data
        customer: a customer's id
        
    OUTPUT:
        ranges: a list of timeranges when an offer was active (at least viewed till end/completion)
        
    """
    # get combinations of time viewed and processed (max of completed and ended)
    ranges = offer_df[(offer_df['person'] == customer) &
                       (offer_df['event_viewed'] == 1)][['time_viewed', 'time_processed']].values.astype(np.int64)
    
    # put them into list of tuples
    ranges = list(map(tuple, ranges))
    return ranges

def check_overlaps(range_list):
    """
    for a list of timeranges when an offer was running, check for overlapping offers
    
    INPUT:
        range_list: a list of timeranges when an offer was active (at least viewed till end/completion)
        
    OUTPUT:
        overlaps: a list of timeranges when at least two offers were active
        
    """
    overlaps = []
    # for each two-element combination, check if the timeranges overlapped for at least one day
    # if yes, add both timeranges to the overlap list
    for subset in combinations(range_list, 2):
#         potential = range(max(subset[0][0], subset[1][0]), min(subset[0][-1], subset[1][-1]))
        potential = max(subset[0][0], subset[1][0]) - min(subset[0][-1], subset[1][-1])
#         if len(potential) >= 0:
        if potential <= 0:
            overlaps.append(subset[0])
            overlaps.append(subset[1])
    return overlaps
    
def union_ranges(range_list):
    """
    a helper function to union overlapping or consecutive timeranges when an offer was running
    
    INPUT:
        range_list: a list of timeranges when an offer was active (at least viewed till end/completion)
        
    OUTPUT:
        ranges_union: a list of timeranges with two (or more) offers running in parallel or consecutively unioned
        
    """
    # get the union of ranges to determine the actual gaps
    ranges_union = []
    for begin,end in sorted(range_list):
        if ranges_union and ranges_union[-1][1] >= begin - 1:
            ranges_union[-1][1] = max(ranges_union[-1][1], end)
        else:
            ranges_union.append([begin, end])
    return ranges_union

def range_gaps(a, b, r):
    """
    define timeranges with no offer running
    
    INPUT:
        a: parameter used for chaining ranges - should be the minimum timestamp of the dataset
        b: parameter used for chaining ranges - should be the maximum timestamp of the dataset
        r: list of unioned timeranges when at least one offer was running
        
    OUTPUT:
        gap_list: a list of timeranges when no offer was running
        
    """
    ranges = union_ranges(r)
    ranges = sorted(ranges)
    flat = chain((a-1,), chain.from_iterable(ranges), (b+1,))
    gap_list = [[x+1, y-1] for x, y in zip(flat, flat) if x+1 < y]
    return gap_list

def create_dictionary_offers(row, summed):
    """
    create a final df row with a single offer running from viewing till end/completion
    
    INPUT:
        row: one row of processed offer dataframe
        summed: total of transactions done during the validity period
        
    OUTPUT:
        dicting: a dictionary with a customer id, how long the offer was valid, 
                 the offer id and the transaction sum
        
    """
    dicting = {'person': row.person,
               'duration_h': 1 + row.time_processed - row.time_viewed,
              'offer_id': row.offer_id,
              'transaction_sum': summed}
    return dicting

def create_dictionary_no_offers(row, summed, timeframe):
    """
    create a final df row with no offer running
    
    INPUT:
        row: one row of processed offer dataframe
        summed: total of transactions done during no offer period
        
    OUTPUT:
        dicting: a dictionary with a customer id, flag that there was no offer,
                 duration of the gap, and the transaction sum during the gap
        
    """
    dicting = {'person': row.person,
               'duration_h': 1 + timeframe[1] - timeframe[0],
              'offer_id': 0,
              'transaction_sum': summed}
    return dicting
    

def combine_transactional_data(offer_df, transactional_df):
    """
    the function populates a dictionary with offer and transactional data combined.
    
    INPUT:
        offer_df: the DataFrame object with processed offer data
        transactional_df: the DataFrame object with transactional data
        
    OUTPUT:
        data_dictionary: a dictionary of dictionaries that contains data required for the final dataframe
    """
    
    # defining an empty dictionary and an index to be incremented for each row
    data_dictionary = {}
    index = 0
    
    # looping through each customer's history
    for player in offer_df['person'].unique():
        print(player)
        ranges = get_ranges(offer_df, player)
        overlaps = check_overlaps(ranges)
        gaps = range_gaps(0,714,ranges)
        # and through each row this customer has
        for row in offer_df[offer_df['person'] == player].itertuples():
            if row.event_viewed == 0:
                continue 
            if (int(row.time_viewed), int(row.time_processed)) in overlaps:
                continue
            else: 
                # aggregating transactions
                transaction_sum = transactional_df[(transactional_df['person'] == player) &
                                       (transactional_df['time'] >= row.time_viewed) &
                                       (transactional_df['time'] <= row.time_processed)]['amount'].sum()
                data_dictionary[index] = create_dictionary_offers(row, transaction_sum)
                index += 1
        for elem in gaps:
            transaction_sum = transactional_df[(transactional_df['person'] == player) &
                                       (transactional_df['time'] >= elem[0]) &
                                       (transactional_df['time'] <= elem[1])]['amount'].sum()
            data_dictionary[index] = create_dictionary_no_offers(row, transaction_sum, elem)              
            index += 1
    return data_dictionary 


def get_dataset(offer_df, transactional_df, df_po, df_pr, filename, save=False):
    """
    the function populates a dictionary with offer and transactional data combined.
    
    INPUT:
        offer_df: DataFrame object with processed offer data
        transactional_df: DataFrame object with transactional data
        filename: a file name chosen for the file in case of a save
        save: when save, overwrites and/or saves the dataframe to the file with a given name
        
    OUTPUT:
        data_dictionary: a dictionary of dictionaries that contains data required for the final dataframe
    """
    data_dict_final = combine_transactional_data(offer_df, transactional_df)
    data_df = pd.DataFrame.from_dict(data_dict_final, orient='index')
    data_df = process_final_df(data_df, df_po, df_pr)
    if save:
        data_df.to_csv(filename + '.csv', index=False)
    return data_df

def create_final_offer_and_trans_dataset(t_df, p_df, qry):
    """
    the function creates the offer and transactional dataframes
    
    INPUT:
        t_df: DataFrame object containing the transcript data
        p_df: DataFrame object containing the portfolio data
        qry: SQL query to combine dataframes
        
    OUTPUT:
        df_offer_final: DataFrame object containing combined and filtered offer data
        df_transaction_final: DataFrame object containing transactional data
    """
    df_viewed = generate_offer_v_df(t_df)
    df_received = generate_offer_r_df(t_df, p_df)
    df_completed = generate_offer_c_df(t_df)
    joined_df = perform_sql_join(df_received, df_completed, df_viewed, qry)
    df_offer_final = filter_out_rows(joined_df)
    df_offer_final = process_offer_df(df_offer_final)

    # assertion needed to make sure the newly created dataframe
    # does not have more or less offers received than the raw dataset
    assert len(df_received) == len(df_offer_final)
    df_transaction_final = get_transaction_df(t_df)
    return df_offer_final, df_transaction_final

# def load_data(name, df_o, df_t, df_po, df_pr):
#     my_file = Path(name + ".csv")
#     if not my_file.is_file():
#         final_df = get_dataset(df_o, df_t, df_po, df_pr, name, True)
#     else:
#         final_df = pd.read_csv(name + ".csv")
#     return final_df


def process_final_df(df_f, df_po, df_pr):
    df_f = df_f.groupby(['person', 'offer_id']).sum().reset_index()
    df_f = df_f.merge(df_po, how='left', left_on='offer_id', right_on='id')
    df_f = df_f.drop(['channels', 'duration', 'offer_type', 'id'], axis=1)
    df_f[['channels_int', 'offer_type_int']] = df_f[['channels_int', 'offer_type_int']].fillna(999)
    df_f = df_f.fillna(0)
    df_f = df_f[df_f['person'].isin(df_pr['id'].unique())].reset_index(drop=True)
    df_f = df_f.merge(df_pr, how='left', left_on='person', right_on = 'id')
    df_f['hourly_sum'] = df_f['transaction_sum'] / df_f['duration_h_x']
    df_f = df_f.drop(['duration_h_x', 'transaction_sum',
                         'id', 'became_member_on'], axis=1)
    return df_f
    
    