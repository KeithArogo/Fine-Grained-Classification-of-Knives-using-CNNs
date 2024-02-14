# 1 Introduction

In recent years, the number of crimes and offences by sharp knives has increased all over the world. In
the year ending March 2022, there were around 45,000 offences involving a knife or sharp instrument in
England and Wales. This was 9% higher than in 2020/21 and 34% higher than in 2010/11. There is critical
need to develop AI weapon analysis system to allow officers to identify the make, model and sellers of
knives, simply by taking a photo. Current methods of knife recognition and labeling, although functional,
have been plagued by inefficiencies due to the immense strain on public resources and the potential for
human error, exacerbated by the UK’s expanding digital footprint. This project develops a deep
learning based system to classify knife images into different categories (classes).

# Parameters
The hyperparameters are present in config.py.

# Trainning
Run train.py to train the model on knife dataset. The default model is EfficientNet-b0

# Testing
Run test.py to test the model on test dataset.

# Note
The dataset was not uploaded due to ownership. 
Demontrations can be given on request.
The csv files contain folder names and directory locations, this can help in understanding how the un-uploaded data interacts with the models.
