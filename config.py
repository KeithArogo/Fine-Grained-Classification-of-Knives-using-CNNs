class DefaultConfigs(object):
    n_classes = 192  ## number of classes
    img_weight = 224  ## image width
    img_height = 224  ## image height
    batch_size = 32 ## 16 - original batch size.. Best - 10
    epochs = 20    ## epochs original - 20 epochs.
    learning_rate = 0.0001  # original - 0.00005 # best - 0.0001 learning rate
config = DefaultConfigs()
