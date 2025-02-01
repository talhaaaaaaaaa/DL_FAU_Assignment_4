import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import src_to_implement.model as model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
df = pd.read_csv('data.csv')  # Adjust the path to your data.csv file
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_dataset = ChallengeDataset(train_df)
val_dataset = ChallengeDataset(val_df)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


# create an instance of our ResNet model
# TODO
resnet_model = model.ResNet() 

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion = t.nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.001)

trainer = Trainer(
    model=resnet_model,
    crit=criterion,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=False,  # Set to True if GPU is available
    early_stopping_patience=5
)


res = trainer.fit(epochs=5)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')