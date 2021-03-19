Objective
The objective for this project is to predict class masks for one test image located at test file

It looks like 
![Screenshot](https://github.com/LiCao-Li/U_Net_Modeling/blob/main/test_image.png)

Instead of simple logloss (also known as binary crossentropy) we use weighted logloss as loss function.
It is defined as:

![Screenshot](https://github.com/LiCao-Li/U_Net_Modeling/blob/main/weight_loss.png)

where  ℎ  and  𝑤  are height and width of original image,  𝑦pred[𝑐,𝑖,𝑗]  are predicted probabilities of pixel  [𝑖,𝑗]  belonging to class  𝑐 ,  𝑦true[𝑐,𝑖,𝑗]  is true indicator ( 1  or  0 ) and  weight  is a vector of class weights: CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]


I used sbatch file to fit the U_net model on RCC GPU resource and the goal for the project is to achieve weighted logloss of 0.29 or better. I achieved weighted logloss of 0.25