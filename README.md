Objective
The objective for this project is to predict class masks for one test image located at test file

It looks like 
![Screenshot](https://github.com/LiCao-Li/U_Net_Modeling/blob/main/test_image.png)

Instead of simple logloss (also known as binary crossentropy) we use weighted logloss as loss function.
It is defined as:

![Screenshot](https://github.com/LiCao-Li/U_Net_Modeling/blob/main/weight_loss.png)

where  β  and  π€  are height and width of original image,  π¦pred[π,π,π]  are predicted probabilities of pixel  [π,π]  belonging to class  π ,  π¦true[π,π,π]  is true indicator ( 1  or  0 ) and  weight  is a vector of class weights: CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]


I used sbatch file to fit the U_net model on RCC GPU resource and I achieved weighted logloss of 0.25.