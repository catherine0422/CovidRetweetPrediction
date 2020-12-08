# Covid Retweet Prediction
Retweet prediction of the tweets about Covid 19.
Project of INF554


关于Version 1:

按照论文，我们要跑14个模型（1个lrrf，10个nnrf，1个xDeepFM，2个DeepFM），参数可以在https://github.com/parklize/cikm2020-analyticup 里找到。

值得注意的是：每个模型训练后都会将模型保存，以供test时用，因此在训练模型之前需要修改保存的名称。我加入了一个log文件，以记录每个模型的保存位置，和对应的参数。所以做训练之前，需要修改：1.保存名称（2个，一个对应回归模型，一个对应随机森林模型）；2.参数和log文件保存的参数，以供查询。

在做test时，需要先载入之前保存的模型，test的结果（MAE分数）会保存在output文件夹里。

最终的MAE分数由这14个模型平均得到。

注：若选择的feature不包括"timeseg", "day_of_week"，需要修改train_deepfmrf，train_xdeepfmrf和fmrf_predict和函数里的一小部分，这些都已经在代码里注明了。
