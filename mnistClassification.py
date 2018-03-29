import torchvision
import torch
import argparse
import ConvStackClassifier
import numpy
import os

print ("mnistClassification.py")

parser = argparse.ArgumentParser()
parser.add_argument('mnistBaseDirectory', help='The directory containing processed/training.pt and processed/test.pt')
parser.add_argument('--mustDownload', action='store_true', help='Download the dataset')
parser.add_argument('--maximumNumberOfTrainingImages', help='The maximum number of training images to load (default: all)', type=int, default=0)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate', type=float, default=0.001)
parser.add_argument('--momentum', help='The learning momentum', type=float, default=0.9)
parser.add_argument('--dropoutRatio', help='The dropout ratio', type=float, default=0.5)
parser.add_argument('--saveDirectory', help='The directory where the files will be saved', default='/tmp')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# -------------------- Utilities -------------------------------

def MinibatchIndices(numberOfSamples, minibatchSize):
    shuffledList = numpy.arange(numberOfSamples)
    numpy.random.shuffle(shuffledList)
    minibatchesIndicesList = []
    numberOfWholeLists = int(numberOfSamples / minibatchSize)
    for wholeListNdx in range(numberOfWholeLists):
        minibatchIndices = shuffledList[ wholeListNdx * minibatchSize : (wholeListNdx + 1) * minibatchSize ]
        minibatchesIndicesList.append(minibatchIndices)
    # Add the last incomplete minibatch
    if numberOfWholeLists * minibatchSize < numberOfSamples:
        lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
        minibatchesIndicesList.append(lastMinibatchIndices)
    return minibatchesIndicesList

# ----------------------------------------------------------------



trainMnistDataset = torchvision.datasets.MNIST(args.mnistBaseDirectory, train=True, download=args.mustDownload)
validationMnistDataset = torchvision.datasets.MNIST(args.mnistBaseDirectory, train=False, download=args.mustDownload)

if args.maximumNumberOfTrainingImages <= 0 or args.maximumNumberOfTrainingImages > trainMnistDataset.__len__():
    args.maximumNumberOfTrainingImages = trainMnistDataset.__len__()
numberOfValidationImages = int( validationMnistDataset.__len__() * args.maximumNumberOfTrainingImages / trainMnistDataset.__len__() )
if numberOfValidationImages > validationMnistDataset.__len__():
    numberOfValidationImages = validationMnistDataset.__len__()

# Put the data in tensors
# Get the image size
image0, target0 = trainMnistDataset.__getitem__(0)
#image0.show()
imgSize = image0.size # (width, height)
print ("imgSize = {}".format(imgSize))
trainImageTensor = torch.Tensor(args.maximumNumberOfTrainingImages, 1, imgSize[1], imgSize[0])
trainLabelTensor = torch.LongTensor(args.maximumNumberOfTrainingImages)

validationImageTensor = torch.Tensor(numberOfValidationImages, 1, imgSize[1], imgSize[0])
validationLabelTensor = torch.LongTensor(numberOfValidationImages)

# Image transformations
imageTransformations = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5, ), (0.3, ))
                                                       ] )

for trainExampleNdx in range(args.maximumNumberOfTrainingImages):
    image, classIndex = trainMnistDataset.__getitem__(trainExampleNdx)
    # The MNIST digits are white on black background
    imageTensor = imageTransformations(image)
    imageTensor.unsqueeze_(0)  # imageTensor.shape: torch.Size([1, 28, 28]) -> torch.Size([1, 1, 28, 28])
    # Put it in the tensor
    trainImageTensor[trainExampleNdx] = imageTensor
    trainLabelTensor[trainExampleNdx] = classIndex

for validationExampleNdx in range(numberOfValidationImages):
    image, classIndex = validationMnistDataset.__getitem__(validationExampleNdx)
    imageTensor = imageTransformations(image)
    imageTensor.unsqueeze_(0)  # imageTensor.shape: torch.Size([1, 28, 28]) -> torch.Size([1, 1, 28, 28])
    # Put it in the tensor
    validationImageTensor[validationExampleNdx] = imageTensor
    validationLabelTensor[validationExampleNdx] = classIndex

# Create a neural network
numberOfConvolutions_kernelSize_pooling_List = []
for layerNdx in range(3): #args.numberOfConvolutionLayers):
    numberOfConvolutions_kernelSize_pooling_List.append( (32, 7, 2) )
    #numberOfConvolutionKernelsList.append(32)#args.numberOfKernelsPerLayer)
    #maxPoolingKernelList.append(2)

#neuralNet = ConvStackClassifier.NeuralNet(numberOfConvolutionKernelsList, maxPoolingKernelList,
neuralNet = ConvStackClassifier.NeuralNet(numberOfConvolutions_kernelSize_pooling_List,
                                          1, 10, imgSize[0],
                                          args.dropoutRatio)
print("neuralNet.structure = {}".format(neuralNet.structure))
if args.cuda:
    neuralNet.cuda() # Move to GPU
    validationImageTensor = validationImageTensor.cuda()
    validationLabelTensor = validationLabelTensor.cuda()

optimizer = torch.optim.SGD(neuralNet.parameters(), lr=args.learningRate, momentum=args.momentum)
lossFunction = torch.nn.NLLLoss()

minibatchSize = 64
minibatchIndicesListList = MinibatchIndices(args.maximumNumberOfTrainingImages, minibatchSize)
trainingDataFilepath = os.path.join(args.saveDirectory, 'trainingEpochs.csv')
trainingDataFile = open(trainingDataFilepath, "w")
trainingDataFile.write('epoch,averageTrainLoss,validationLoss\n')

for epoch in range(200):
    averageTrainLoss = 0
    for minibatchListNdx in range(len(minibatchIndicesListList)):
        minibatchIndicesList = minibatchIndicesListList[minibatchListNdx]
        thisMinibatchSize = len(minibatchIndicesList)

        minibatchInputImagesTensor = torch.autograd.Variable(
            torch.index_select(trainImageTensor, 0, torch.LongTensor(minibatchIndicesList)))
        minibatchTargetOutputTensor = torch.autograd.Variable(
            torch.index_select(trainLabelTensor, 0, torch.LongTensor(minibatchIndicesList)))
        if args.cuda:
            minibatchInputImagesTensor = minibatchInputImagesTensor.cuda()
            minibatchTargetOutputTensor = minibatchTargetOutputTensor.cuda()
            #neuralNet.cuda()

        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        actualOutput = neuralNet(minibatchInputImagesTensor)
        # Loss
        """targetOutputShape = minibatchTargetOutputTensor.data.shape
        actualOutputShape = actualOutput.data.shape
        print("targetOutputShape = {}; actualOutputShape = {}".format(targetOutputShape, actualOutputShape))
        """
        loss = lossFunction(actualOutput, minibatchTargetOutputTensor)
        # if minibatchNdx == 0:
        #    print("Train loss = {}".format(loss.data[0]))

        # Backward pass
        loss.backward()
        # Parameters update
        optimizer.step()

        averageTrainLoss += loss.data[0]

    averageTrainLoss = averageTrainLoss / len(minibatchIndicesListList)

    # Validation loss
    validationOutput = neuralNet( torch.autograd.Variable(validationImageTensor) )
    validationLoss = lossFunction(torch.nn.functional.log_softmax(validationOutput), torch.autograd.Variable(
        validationLabelTensor) )

    print("Epoch {}: Average train loss = {}; validationLoss = {}".format(epoch, averageTrainLoss, validationLoss.data[0]))
    neuralNet.Save(args.saveDirectory, str(validationLoss.data[0]))
    trainingDataFile.write("{},{},{}\n".format(epoch, averageTrainLoss, validationLoss.data[0]) )

trainingDataFile.close()