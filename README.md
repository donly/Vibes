# Vibes

Core ML 是在iOS 11发布的。

将机器学习模型引入App的方法，实现在iOS设备上直接使用 AI 相关的功能，比如物体检测。

iOS 13 在 Core ML 3 中增加了on-device training（基于设备的模型训练）功能，为框架增加了基于设备进行模型个性化微调能力。


人工智能Artificial Intelligence（AI）定义为：以编程方式添加到机器上以模仿人类的行动和思想的能力。
机器学习Machine Learning（ML）是人工智能的子集，训练机器执行某些特定任务。例如，你可以使用 ML 来训练机器识别图像中的猫，或将文字从一种语言翻译成另一种语言。
深度学习Deep Learning是一种机器训练方法。这种技术模仿人脑，由组织在网络中的神经元组成。深度学习从提供的数据中训练出一个人工神经网络。

苹果将模型 (Model) 定义为“将机器学习算法应用于一组训练数据的结果”。

把模型看作是一个函数，它接受一个输入，对给定的输入进行特定的操作，使其达到最佳效果，比如学习，然后进行预测和分类，并产生合适的输出。
用标记的数据进行训练被称为监督学习（supervised learning）。你需要大量的优质数据来建立一个优质模型。什么是 优质 ？优质数据要尽可能全面，因为最终建立的模型全部依赖于喂给机器的数据。

比如，如果你想让你的模型识别所有的猫，但只给它提供一个特定的品种，它可能会不认识在这些品种之外的猫。用残缺的数据进行训练会导致不想要的结果。

训练过程是计算密集型的，通常在服务器上完成。凭借其并行计算能力，使用 GPU 通常会加快训练的速度。
一旦训练完成，你可以将你的模型部署到生产中，在真实世界的数据上运行预测或推理。

￼

预测推理Inference并不像训练那样需要计算。然而在过去，移动 App 必须远程调用服务器接口才能进行模型推理。现在，移动芯片性能的进步为设备上的推理(on-device inference)打开了大门。其好处包括减少延迟，减少对网络的依赖和改善隐私。但是，由于推理运算提高了硬件负载，应用程序大小会增加，推理时电池消耗也会有明显的提升。


Vision框架提供了再图像或视频上执行计算机视觉算法的高阶API封装。Vision可以使用苹果提供的内置模型或者自定义的Core ML模型对图像进行分类（classify）。

Core ML是建立在低级别基元(lower-level primitives)：BNNS加速 和 Metal高效着色器之上的。
￼


可更新的模型是一个标记为“可更新”的Core ML模型，你也可以将你自己训练的模型定义为可更新的。

K最近邻分类(k-NN)算法，k-NN(k-Nearest Neighbors)

它通过比较特征向量 (feature vectors) 来达到想要的结果，一个特征向量包含描述一个物体特征的关键信息，比如使用特征向量R、G、B来表示RGB颜色。

￼

**k-NN **模型简单而又迅速，不需要很多数据就可以训练。但是随着样例数据越多，它的性能也会变得越慢。
k-NN 是 Core ML 支持训练的模型类型之一

