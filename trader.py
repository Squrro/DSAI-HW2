# You can write code above the if-main block.

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.

    import pandas as pd
    import math
    import pandas_datareader as web
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint


    ##df = pd.read_csv('./training.csv')
    import csv
    data = []
    i = 0
    with open(args.training,'r',newline='') as csvfile:
        df = csv.reader(csvfile)
        for row in df :
            data.append([i,row[0]])
            i = i +1
    dataset = data
    train_data_len = math.ceil(len(dataset))
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(dataset)
    scaled_data = scaler.transform(dataset)
    train_data = scaled_data[0:train_data_len,:]
    
    x_train = []
    y_train = []
    for i in range(100,len(train_data)):
        x_train.append(train_data[i-100:i,0])
        y_train.append(train_data[i:0,0])       
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    
    model = Sequential()
    model.add(LSTM(50,return_sequences = True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    model.fit(x_train,y_train,batch_size=50,epochs=2)
    prediction = model.predict(x_train)
    #prediction = scaler.inverse_transform(prediction)
    
    #############################
    count = -1
    tst = []
    state = 0
    with open(args.testing,'r',newline='') as csvfile:
        with open(args.output, "w") as output_file:
            writer = csv.writer(output_file)
            test = csv.reader(csvfile)
            for row in test :
                tst.append([count,row[0]])
                count = count +1
                if count-1 < 0:
                    continue
                # We will perform your action as the open price in the next day.
                if prediction[count]-prediction[count-1] >= 0:
                    if state == 0  :
                        writer.writerow("1")
                        state = 1
                    elif state == 1:
                        writer.writerow("0")
                        state = 1
                elif prediction[count]-prediction[count-1] < 0:
                    if state == 0 :
                        writer.writerow("0")
                        state = 0
                    elif state == 1:
                        state = 0
                        writer.writerow("-1")            

            #for row in range(count-1):

                # this is your option, you can leave it empty.
                #trader.re_training()
