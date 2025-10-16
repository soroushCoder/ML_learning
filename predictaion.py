# Minimal RNN with explicit matrices (like your image)
# pip install tensorflow

import numpy as np
import tensorflow as tf

# -----------------------------
# 1) Toy monthly gold prices
#    (oldest -> newest). Replace with your data.
# -----------------------------
prices = np.array([
    1800, 1815, 1822, 1830, 1842, 1850, 1863, 1872, 1880, 1895, 1902, 1910,
    1905, 1912, 1920, 1935, 1942, 1950, 1965, 1970, 1982, 1990, 2005, 2010,
    2002, 2015, 2022, 2035, 2048, 2055, 2060, 2068, 2075, 2088, 2095, 2102
], dtype=np.float32)

WINDOW = 12   # use last 12 months to predict next

# -----------------------------
# 2) Very simple scaling to [0,1]
# -----------------------------
p_min, p_max = prices.min(), prices.max()
prices_s = (prices - p_min) / (p_max - p_min + 1e-9)

# Build (X, y): X[i] -> 12 months, y[i] -> next month
X, y = [], []
for i in range(len(prices_s) - WINDOW):
    X.append(prices_s[i:i+WINDOW])
    y.append(prices_s[i+WINDOW])
X = np.array(X, dtype=np.float32)             # (N, 12)
y = np.array(y, dtype=np.float32).reshape(-1, 1)  # (N, 1)

# Last window to predict the NEXT (unseen) month
X_train, y_train = X[:-1], y[:-1]
X_last = X[-1:]                                # shape (1, 12)

# -----------------------------
# 3) Custom RNN cell (like the slide)
# -----------------------------
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super().__init__()
        # Initialize weight matrices
        self.W_xh = self.add_weight(shape=(rnn_units, input_dim),
                                    initializer="glorot_uniform", name="W_xh")
        self.W_hh = self.add_weight(shape=(rnn_units, rnn_units),
                                    initializer="orthogonal", name="W_hh")
        self.W_hy = self.add_weight(shape=(output_dim, rnn_units),
                                    initializer="glorot_uniform", name="W_hy")
        # Hidden state placeholder (filled per sequence)
        self.rnn_units = rnn_units

    def reset_state(self, batch_size):
        # h_0 = zeros for each sequence in the batch
        self.h = tf.zeros((batch_size, self.rnn_units, 1), dtype=tf.float32)

    def step(self, x_t):
        # x_t: (batch, input_dim, 1)
        # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)
        h_prev = self.h  # (batch, rnn, 1)

        # Reshape for proper broadcasting: (batch, rnn, 1) -> (batch, rnn)
        h_prev_2d = tf.squeeze(h_prev, axis=-1)  # (batch, rnn)
        x_t_2d = tf.squeeze(x_t, axis=-1)  # (batch, input_dim)

        # Matrix multiplications: W @ x where W is (out, in) and x is (batch, in)
        whh_h = tf.matmul(h_prev_2d, self.W_hh, transpose_b=True)  # (batch, rnn)
        wxh_x = tf.matmul(x_t_2d, self.W_xh, transpose_b=True)  # (batch, rnn)

        # Update hidden state
        h_new = tf.math.tanh(whh_h + wxh_x)  # (batch, rnn)
        self.h = tf.expand_dims(h_new, axis=-1)  # (batch, rnn, 1)

        # y_t = W_hy @ h_t  -> (batch, output_dim)
        y_t = tf.matmul(h_new, self.W_hy, transpose_b=True)  # (batch, output_dim)
        y_t = tf.expand_dims(y_t, axis=-1)  # (batch, output_dim, 1)

        return y_t  # (batch, output_dim, 1)

# -----------------------------
# 4) Tiny training loop
# -----------------------------
INPUT_DIM = 1        # each month is a single value
HIDDEN_UNITS = 8     # n_h
OUTPUT_DIM = 1       # predict one number: next month

cell = MyRNNCell(HIDDEN_UNITS, INPUT_DIM, OUTPUT_DIM)
opt = tf.keras.optimizers.Adam(1e-2)

def run_sequence(x_seq):
    """
    x_seq: (batch, 12) -> returns last output (batch,1)
    We feed one month at a time, exactly like the diagram.
    """
    batch = tf.shape(x_seq)[0]
    cell.reset_state(batch)

    # (batch, 12, 1, 1)
    x_seq = tf.reshape(x_seq, (batch, WINDOW, 1, 1))
    for t in range(WINDOW):
        y_t = cell.step(x_seq[:, t])   # (batch, 1, 1)
    return tf.reshape(y_t, (batch, 1)) # last step output

@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        pred = run_sequence(xb)
        loss = tf.reduce_mean((pred - yb)**2)
    grads = tape.gradient(loss, cell.trainable_variables)
    opt.apply_gradients(zip(grads, cell.trainable_variables))
    return loss

# Train a few epochs (small, for learning)
Xb = tf.convert_to_tensor(X_train, dtype=tf.float32)
yb = tf.convert_to_tensor(y_train, dtype=tf.float32)
for epoch in range(300):
    loss = train_step(Xb, yb)
    if (epoch+1) % 100 == 0:
        tf.print("epoch", epoch+1, "loss", loss)

# -----------------------------
# 5) Predict the next month
# -----------------------------
X_last_tf = tf.convert_to_tensor(X_last, dtype=tf.float32)
pred_scaled = run_sequence(X_last_tf).numpy()[0,0]
pred_price = pred_scaled * (p_max - p_min) + p_min

print(f"\nPredicted NEXT month's gold price: {pred_price:.2f}")
