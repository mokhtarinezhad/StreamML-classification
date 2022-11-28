from keras.models import Model
from mypackage.nn.conv import FCHeadNet
from mypackage.nn import BaseModel

class CreateModel(object):
    
    def __init__(self,basemodel,height,width,channels,nbClasses, stractivation):
        self.basemodel = basemodel
        self.height = height
        self.width = width
        self.channels = channels
        self.nbClasses = nbClasses
        self.stractivation = stractivation

    def GetModelStructure(self):
        # load the specific network, ensuring the head FC layer sets are left off and
        # load pre-training on ImageNet weights on the specific model        
        self.baseModel = BaseModel(self.basemodel,self.height,self.width,self.channels).basemodelselect()
    
        # initialize the new head of the network, a set of FC layers
        # followed by a softmax classifier
        headModel = FCHeadNet.build(self.baseModel, self.nbClasses, self.stractivation, 256)     
        
        # place the head FC model on top of the base model -- this will
        # become the actual model we will train
        model = Model(inputs=self.baseModel.input, outputs=headModel)
        
        self.freeze_layers()
        
        return model
    
    def freeze_layers(self):
        # loop over all layers in the base model and freeze them so they
        # will *not* be updated during the training process
        for layer in self.baseModel.layers:
            layer.trainable = False
            
class CreateModelFinetuneVGG16(CreateModel):
    def __init__(self,basemodel,height,width,channels,nbClasses, stractivation):
        CreateModel.__init__(self,basemodel,height,width,channels,nbClasses, stractivation)
        
        
    def freeze_layers(self):
        for layer in self.baseModel.layers[:15]:
            layer.trainable = False
        # now that the head FC layers have been trained/initialized, lets
        # unfreeze the final set of CONV layers and make them trainable
        for layer in self.baseModel.layers[15:]:
            layer.trainable = True  
            
class CreateModelFinetuneVGG19(CreateModel):
    def __init__(self,basemodel,height,width,channels,nbClasses, stractivation):
        CreateModel.__init__(self,basemodel,height,width,channels,nbClasses, stractivation)
        
    def freeze_layers(self):
        for layer in self.baseModel.layers[:17]:
            layer.trainable = False
        # now that the head FC layers have been trained/initialized, lets
        # unfreeze the final set of CONV layers and make them trainable
        for layer in self.baseModel.layers[17:]:
            layer.trainable = True
            
class CreateModelFinetuneInceptionV3(CreateModel):
    def __init__(self,basemodel,height,width,channels,nbClasses, stractivation):
        CreateModel.__init__(self,basemodel,height,width,channels,nbClasses, stractivation)
        
    def freeze_layers(self):
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 279 layers and unfreeze the rest:
        for layer in self.baseModel.layers[:279]:
            layer.trainable = False
        for layer in self.baseModel.layers[279:]:
            layer.trainable = True
            
class CreateModelFinetuneXception(CreateModel):
    def __init__(self,basemodel,height,width,channels,nbClasses, stractivation):
        CreateModel.__init__(self,basemodel,height,width,channels,nbClasses, stractivation)
        
    def freeze_layers(self):
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 105 layers and unfreeze the rest:
        for layer in self.baseModel.layers[:105]:
            layer.trainable = False
        for layer in self.baseModel.layers[105:]:
            layer.trainable = True