"""
Predicting P2P investors' future investment features with RNN
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras import backend
from keras.layers.core import Dense, Dropout
import keras.layers.recurrent as recurrent_layer
from keras.callbacks import EarlyStopping


features_labels = [
    "credit_score",
    "category",
    "duration",
    "auction_format",
    "prosper_score",
    "is_home_owner",
    "same_BL_rate",
    "title_length",
    "listing_number",
    "lender_rate",
    "debt_to_income_ratio",
    "borrower_rate",
    "borrower_max_rate",
    "bid_count",
    "amount_requested"
]

def load_data(file_name="./data/"):
    """
    load data from raw feature file to pandas dataframe structure
    """
    with open(file_name) as input_file:
        all_user_records = []
        for line in input_file:
            user_id = line.split('\t')[0]
            user_all_features = line.split('\t')[1:]
            for time_step, user_each_time_feature in enumerate(user_all_features):
                user_onetime_features = user_each_time_feature.split('|')
                user_feature_dict = {}
                user_feature_dict['user_id'] = user_id
                user_feature_dict['time_step'] = time_step
                for feature_cat_id, user_onetime_each_feature in enumerate(user_onetime_features):
                    user_feature_dict[features_labels[feature_cat_id]] = np.array(user_onetime_each_feature.split(';'))
            all_user_records.append(user_feature_dict)
    
    df = pd.DataFrame.from_records(all_user_records)
    return df


def create_X_Y(df):
    """
    create training X and Y from pandas dataframe data
    """
    current_user_id = df.loc[0]['user_id']
    X = []
    X_instance = []
    for _, df_rec in df.iterrows:
        tmp_user_id = df_rec['user_id']
        if current_user_id != tmp_user_id:
            X.append(X_instance)
            current_user_id = tmp_user_id
            X_instance = []
        X_onetime_feature = None
        for feature_label in features_labels:
            each_feature = df_rec[feature_label]
            # normalize
            norm_each_feature = np.reshape(normalize(each_feature.reshape(1, -1), norm='l1', axis=1), (-1, ))
            if X_onetime_feature is None:
                X_onetime_feature = norm_each_feature
            else:
                X_onetime_feature = np.concatenate((X_onetime_feature, norm_each_feature))
        X_instance.append(X_onetime_feature)
    X.append(X_instance)  # for the last user
    X = np.array(X)  # change to numpy array

    train_X = X[:, :-1, :]
    train_Y = X[:, -1, :]

    return train_X, train_Y


def loss_invest_vec_mse(y_true, y_pred):
    """
    loss function used to optimize the learning process
    mean squared error across the whole investment vector
    """
    return backend.mean(backend.square(y_true - y_pred))


def build_rnn_structure(input_shape, n_channels=256, rnn_unit=recurrent_layer.GRU, deep_level=5):
    """
    build rnn structure for learing user investment pattern
    [rnn unit (GRU or LSTM or normal RNN) + dense] * deep_level
    """
    our_model = Sequential()
    our_model.add(rnn_unit(units=n_channels, input_shape=input_shape, activation='relu', return_sequences=True))
    our_model.add(Dense(n_channels, activation='relu'))
    for _ in range(deep_level - 2):
        our_model.add(rnn_unit(units=n_channels, activation='relu', return_sequences=True))
        our_model.add(Dense(n_channels, activation='relu'))
    our_model.add(rnn_unit(units=n_channels, activation='relu', return_sequences=False))
    our_model.add(Dense(n_channels, activation='relu'))
    our_model.add(Dropout(0.2))
    our_model.add(Dense(input_shape[-1], activation='softmax'))
    our_model.compile(loss=loss_invest_vec_mse, optimizer='adam')

    return our_model


def load_training_X_Y():
    """
    load training X and Y, if cached in the file system, directly read;
    otherwise create from the original raw feature file
    """
    cache_file_X = 'data/train_X.npy'
    cache_file_Y = 'data/train_Y.npy'

    if Path(cache_file_X).is_file() and Path(cache_file_Y).is_file():
        print("find cached training data!")
        train_X = np.load(cache_file_X)
        train_Y = np.load(cache_file_Y)
    
    else:
        df = load_data()
        train_X, train_Y = create_X_Y(df)
        np.save(cache_file_X, train_X)
        np.save(cache_file_Y, train_Y)

    return train_X, train_Y


def global_mean(X, Y):
    """
    test global mean
    """
    global_mean_Y = np.mean(np.mean(X, axis=0), axis=0)
    mse = np.mean(np.square(Y - global_mean_Y))
    print("global mean's mean squared error: {}".format(mse))
    return global_mean_Y, mse


def personal_mean(X, Y):
    """
    test personal mean
    """
    personal_mean_Y = np.mean(X, axis=1)
    mse = np.mean(np.square(Y - personal_mean_Y))
    print("personal mean's mean squared error: {}".format(mse))
    return personal_mean_Y, mse


def personal_k_nearest_mean(X, Y, k_near):
    """
    test personal mean on nearest k time steps
    """
    X_near_k = X[:, -k_near:, :]
    personal_near_k_mean_Y = np.mean(X_near_k, axis=1)
    mse = np.mean(np.square(Y - personal_near_k_mean_Y))
    print("personal {} nearest mean's mean squared error: {}".format(k_near, mse))
    return personal_near_k_mean_Y, mse


def cross_validataion_rnn(X, Y, rnn_builder=build_rnn_structure, K=5):
    """
    cross validation on RNN
    """
    k_fold = KFold(K)
    Y_pred = np.zeros(Y.shape)
    for train, test in k_fold.split(X, Y):
        rnn_model = rnn_builder((X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        rnn_model.fit(X[train], Y[train], batch_size=32, epochs=20, verbose=1, validation_data=(X[test], Y[test]), callbacks=[early_stop])
        Y_pred[test] = rnn_model.predict(X[test])

    return Y_pred


if __name__ == "__main__":
    p2p_X, p2p_Y = load_training_X_Y()
    global_mean(p2p_X, p2p_Y)
    personal_mean(p2p_X, p2p_Y)
    for k in range(1, 6):
        personal_k_nearest_mean(p2p_X, p2p_Y, k)
    cross_validataion_rnn(p2p_X, p2p_Y)
    