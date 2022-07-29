from MAIN.MODELS.CLASSIFICATION.utils import load_data_tf
import tensorflow as tf

# value = {"Train":train,"Test":test,"encoder": encoder,"output_dim":len(unique_labels_name),"embadding_dim":Config.Embadding_Dim}

config = load_data_tf()

model = tf.keras.layers.Sequential([
        config['encoder'], 
        tf.keras.layers.Embedding(
            input_dim=len(config['encoder'].get_vocabulary()),
            output_dim=config['embadding_dim'],
            mask_zero=True
                                ),
        tf.keras.layers.Bidirectional(
                tf.keras.layers.RNN(16,return_sequences=True)
                                      ),
        tf.keras.layers.Bidirectional(
                tf.keras.layers.RNN(10)
                                      ),
        tf.keras.layers.Dense(10,activation='softmax'),
        tf.keras.layers.Dense(config['output_dim'],activation='softmax')

        ])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
        )