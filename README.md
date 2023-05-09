# Retina-Damage-Detection-
### Objective:
The goal of this project is to utilize conventional neural network to classify Retina Damage with best accuracy possible.
### Requirements:
-numpy
-tensorflow
-matplotlib
### Approach :
-we will load the train,test,validation folders that contains the data and I did some visualization to see the normal and the abnormal batch and we took some samples some to apply preprocessing as I applied some filters like Gussian,poisson,salt and paper 

![Alt text](https://github.com/menna566/Retina-Damage-Detection-/blob/main/photo_6025840907745606851_y.jpg)

## as we see we lost a lot of information so we wont apply these filters on our data 


our next step that we will try to prepare our data and adjust its dimensions and then we will build a conventional neural network model using tensorflow to fit the model on the training set and later we will test it on the test set and compare it with the validation set to analyze if overfitting took place. 

### Results 
-the model accuracy is :0.93

![Alt text]()



![Alt text]()

-we observe that the training loss was closs to the validation loss so i think our model dont have overfitting 
