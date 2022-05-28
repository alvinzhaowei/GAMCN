import tensorflow as tf
import numpy as np

# read traffic values
Traffic = np.load("./data/pems04.npy")


# create one-hot time embedding vectors and divide data into train, val, and test
def data_preprocess(train_ratio, test_ratio):
    # train/val/test
    num_step = Traffic.shape[0]
    time_coding = np.zeros((num_step, 295, 1), dtype="float32")
    s_day = 4
    for i in range(num_step):
        day = (s_day + i // 288) % 7
        time_coding[i][i % 288] = 1
        time_coding[i][288 + day] = 1
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    train = Traffic[: train_steps].astype('float32')
    train_time = time_coding[: train_steps]
    val = Traffic[train_steps:-test_steps].astype('float32')
    val_time = time_coding[train_steps:-test_steps]
    test = Traffic[-test_steps:].astype('float32')
    test_time = time_coding[-test_steps:]
    length = len(train)
    return train, train_time, test, test_time, val, val_time


train_data, train_time, test, test_time, val, val_time = data_preprocess(0.7, 0.2)
# del Traffic


# read adjacent matrix and generate P_forward and P_backward
adj_matrix = "./data/Adj(PEMS04).txt"


def read_adj(file_name):
    count = 0
    f = open(file_name, "r")
    lines = f.readlines()
    n = len(lines)
    n = int(np.sqrt(n))
    adj_m = np.zeros((n, n), dtype="float32")
    for line in lines:
        node1, node2, edge = line[:-1].split()
        edge = float(edge)
        if edge == 0:
            continue
        else:
            count = count + 1
            adj_m[int(node1)][int(node2)] = edge
    return adj_m


adj = read_adj(adj_matrix)
adj = adj - np.diag(np.ones((adj.shape[0],), dtype="float32"))
adj_T = np.transpose(adj)

P_f = np.zeros(adj.shape, dtype="float32")
for i, item in enumerate(np.sum(adj, axis=1)):
    if item == 0:
        continue
    P_f[i] = adj[i] / item

P_b = np.zeros(adj_T.shape, dtype="float32")
for i, item in enumerate(np.sum(adj_T, axis=1)):
    if item == 0:
        continue
    P_b[i] = adj_T[i] / item

# generate input data using a sliding window, and normalize the data
ts_window = 12


def generate_input(data, data_time, m, s):
    t, n = data.shape
    data = (data - m) / s
    window_size = 24
    input_size = t - window_size
    input_fm = np.zeros((input_size, window_size, n, 1), dtype="float32")
    input_tm = np.zeros((input_size, window_size, 295), dtype="float32")
    for i in range(input_size):
        fm = data[i:i + window_size]
        input_fm[i] = fm.reshape(window_size, n, 1)
        input_tm[i] = data_time[i:i + window_size].reshape(window_size, 295)
    return input_tm, input_fm


m_train, s_train = np.mean(train_data), np.std(train_data)
train_tm, train_X_Y = generate_input(train_data, train_time, m_train, s_train)
val_tm, val_X_Y = generate_input(val, val_time, m_train, s_train)
test_tm, test_X_Y = generate_input(test, test_time, m_train, s_train)


# LPGCNLayer
class LPGCNLayer(tf.keras.Model):
    def __init__(self, P_pair, k):
        super(LPGCNLayer, self).__init__()
        self.P_pair = P_pair
        self.k = k

    def build(self, input_shape):
        self.shape = input_shape

        self.F = tf.random.uniform(
            shape=(self.shape[-2], self.shape[-2]),
            minval=0, maxval=self.shape[-2], dtype=tf.dtypes.int32, name="F",
        )
        self.w1 = []
        for i in range(self.k):
            self.w1.append(self.add_weight(name='GCNW1_' + str(i),
                                           shape=(self.shape[-1], self.shape[-1]),
                                           initializer='RandomNormal',

                                           ))
        self.w2 = []
        for i in range(self.k):
            self.w2.append(self.add_weight(name='GCNW2_' + str(i),
                                           shape=(self.shape[-1], self.shape[-1]),
                                           initializer='RandomNormal',

                                           ))
        self.w3 = []
        for i in range(self.k):
            self.w3.append(self.add_weight(name='GCNW3_' + str(i),
                                           shape=(self.shape[-1], self.shape[-1]),
                                           initializer='RandomNormal',

                                           ))

    def call(self, feature_matrix, trainable=True):
        # B,N,F
        P_f, P_b = self.P_pair
        Z_f = tf.nn.relu(tf.matmul(feature_matrix, self.w1[0]))
        Z_b = tf.nn.relu(tf.matmul(feature_matrix, self.w2[0]))
        Z_a = tf.nn.relu(tf.matmul(feature_matrix, self.w3[0]))
        T_f = feature_matrix
        T_b = feature_matrix
        T_a = feature_matrix
        F = self.F
        sumF = tf.reduce_sum(F)
        p = tf.divide(F, sumF)
        pi = tf.divide(tf.reduce_sum(F, 0), sumF)
        pj = tf.divide(tf.reduce_sum(F, 1), sumF)
        zero_matrix = tf.zeros((self.shape[-2], self.shape[-2]))
        pi_j = tf.matmul(tf.reshape(pi, (pi.shape[0], 1)), tf.reshape(pj, (1, pj.shape[0])))
        P = tf.math.log(tf.divide(p, pi_j))
        P = tf.cast(P, dtype=tf.float32)
        P = tf.math.maximum(P, zero_matrix)
        for i in range(1, self.k):
            T_f = tf.matmul(P_f, T_f)
            T_b = tf.matmul(P_b, T_b)
            T_a = tf.matmul(P, T_a)
            Z_f = Z_f + tf.nn.relu(tf.matmul(T_f, self.w1[i]))
            Z_b = Z_b + tf.nn.relu(tf.matmul(T_b, self.w2[i]))
            Z_a = Z_a + tf.nn.relu(tf.matmul(T_a, self.w3[i]))
        result = Z_a + Z_f + Z_b
        if trainable:
            result = tf.nn.dropout(result, rate=0.3)
        return result


# multi-path CNN
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        self.convs = [i for i in range(ts_window - 1)]
        self.lrelus = [i for i in range(ts_window - 1)]
        self.conre = [i for i in range(ts_window)]
        for i in range(0, ts_window - 1):
            self.convs[i] = tf.keras.layers.Conv1D(filters=self.shape[-1], kernel_size=i + 2, activation="relu",
                                                   trainable=True)

    def call(self, input_tensor, training):
        self.conre[0] = tf.transpose(input_tensor, perm=[0, 2, 1, 3])
        for i in range(ts_window - 1):
            self.conre[i + 1] = self.convs[i](self.conre[0])
        combine = tf.concat(self.conre, axis=2)
        return combine


# attention block
class Temporal_block(tf.keras.Model):
    def __init__(self, k):
        super(Temporal_block, self).__init__()
        self.k = k

    def build(self, input_shape):
        self.w = self.add_weight(
            name='TBW_' + str(self.k),
            shape=(400, 295, 78),
            initializer='RandomNormal',

        )
        self.b = self.add_weight(
            name='TBB_' + str(self.k),
            shape=(400, 1, 78),
            initializer='RandomNormal',

        )

    def call(self, input_tensor, training):
        input_tensor = tf.reshape(input_tensor, [tf.shape(input_tensor)[0], 1, 1, tf.shape(input_tensor)[1]])
        w = tf.broadcast_to(self.w, [tf.shape(input_tensor)[0], 400, 295, 78])
        b = tf.broadcast_to(self.b, [tf.shape(input_tensor)[0], 400, 1, 78])
        x = tf.add(tf.matmul(input_tensor, w), b)
        return x


class GEN(tf.keras.Model):
    def __init__(self):
        super(GEN, self).__init__(name='')

    def build(self, input_shape):
        self.shape = input_shape
        self.gcn = LPGCNLayer((P_f, P_b), 2)
        self.cnn = CNN()
        self.tb = []
        self.w1 = []
        self.w2 = []
        self.b = []
        for i in range(12):
            self.tb.append(Temporal_block(i))

            self.w1.append(self.add_weight(name='GENW1' + str(i),
                                           shape=(100, 100),
                                           initializer='RandomNormal',

                                           ))
            self.w2.append(self.add_weight(name='GENW2' + str(i),
                                           shape=(100, 100),
                                           initializer='RandomNormal',

                                           ))
            self.b.append(self.add_weight(name='GENB' + str(i),
                                          shape=(100,),
                                          initializer='RandomNormal',

                                          ))
        self.dense1 = tf.keras.layers.Dense(10, activation="relu")
        self.dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.dense3 = tf.keras.layers.Dense(10, activation="relu")
        self.dense4 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=True):
        time_matrix, feature_matrix = inputs
        xb = self.dense1(feature_matrix)
        xb = self.dense2(xb)

        if training:
            xb = tf.nn.dropout(xb, rate=0.3)
        results = []
        gcnx = self.gcn(xb, training=training)
        x = self.cnn(xb, training=training)
        x = tf.transpose(x, perm=[0, 3, 2, 1])
        for i in range(12):
            attention = self.tb[i](time_matrix[:, i, :])
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            re1 = tf.matmul(attention, x)
            re1 = tf.linalg.diag_part(re1)
            re1 = tf.transpose(re1, perm=[0, 2, 1])
            re1 = tf.reshape(re1, [tf.shape(re1)[0], 1, tf.shape(re1)[1], tf.shape(re1)[2]])
            re2 = gcnx[:, i:i + 1, :, :]
            b = tf.broadcast_to(self.b[i], [tf.shape(x)[0], 1, 400, 100])
            z = tf.nn.sigmoid(tf.matmul(re1, self.w1[i]) + tf.matmul(re2, self.w2[i]) + b)
            temp_result = tf.multiply(z, re1) + tf.multiply((1 - z), re2)
            results.append(temp_result)
        x = tf.concat(results, axis=1)
        x = self.dense3(x)
        x = self.dense4(x)

        return x


# train
generator = GEN()
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mae", metrics=["mae"])
train_tm = train_tm[:, 12:, :]
train_X = train_X_Y[:, :12, :]
train_Y = train_X_Y[:, 12:, :]

val_tm = val_tm[:, 12:, :]
val_X = val_X_Y[:, :12, :]
val_Y = val_X_Y[:, 12:, :]


generator.fit(x=[train_tm, train_X], y=train_Y, batch_size=8, epochs=10, validation_data=([val_tm, val_X], val_Y))

# evaluate

test_tm = test_tm[:, 12:, :]
test_X = test_X_Y[:, :12, :]
test_Y = test_X_Y[:, 12:, :]

# predict
predictions = generator.predict([test_tm, test_X], batch_size=8)


# denormalize and calculate the errors
def cal_error(g_t, prediction, p):
    g_t = g_t[:, p, :, :] * s_train + m_train
    prediction = prediction[:, p, :, :] * s_train + m_train
    nzindex = np.nonzero(g_t)
    g_t = g_t[nzindex]
    prediction = prediction[nzindex]
    abe = np.fabs(g_t - prediction)
    loss_3 = np.mean(abe * abe)
    loss_1 = np.mean(abe)
    loss_2 = np.mean(abe / g_t)
    print("MAE:", loss_1)
    print("MAPE:", loss_2)
    print("RMSE:", np.sqrt(loss_3))


# test different time slot
print("15 minutes")
cal_error(test_Y, predictions, 2)
print("30 minutes")
cal_error(test_Y, predictions, 5)
print("60 minutes")
cal_error(test_Y, predictions, 11)
