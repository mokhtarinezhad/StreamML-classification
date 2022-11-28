# USAGE: final code for classification training
# python train_model_folder_trlearning_gpu.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, SGD, Adam, Nadam, Adagrad

#from keras.models import Model
from keras.utils.training_utils import multi_gpu_model

#from mypackage.nn.conv import FCHeadNet
from mypackage.callbacks import MyCbk, MyCbkTimeEstimation
from mypackage.datasets import check_for_dataset_folders
from mypackage.preprocessing import ClassManager, ClassManagerReg, ClassManagerBinary
from mypackage.preprocessing import Preprocessing, ModelPreprocessInput
from mypackage.utils import TestingBlock, TestingBlock_regression, \
TestReporting, TestReporting_multi, TestReporting_regression, \
GeneratorModifier, GeneratorModifier_regression
from mypackage.utils import timer, convert_stringratio
from mypackage.utils import ModelConverter, ModelConverter_regression

#from mypackage.nn import BaseModel

from mypackage.nn import CreateModel

import time
import configparser
import numpy as np
import os
import shutil

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed


# Section used for dynamic memory growth when using GPU
from keras.backend.tensorflow_backend import set_session
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
configtf.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=configtf)
set_session(sess)  # set this TensorFlow session as the default session for Keras


from utils_ext import pDir_models, \
pDir_plots_lc, st2b, st3, TRAIN_PATH, VAL_PATH, TEST_PATH, \
data_prepro_dir, pDir_results, encoder_pickle_fname, pickleSave, \
main_directory, convert_stringtuple, \
error_exit, top_model_warmup_weights_path, \
metric_name_classification, metric_name_cbk_classification, \
stractivation_classification, theloss_rgbtm, theloss_rgbtb, \
metric_name_regression, metric_name_cbk_regression, \
stractivation_regression, theloss_rgbtr, parsetup_fname, pDir_data_dir

# needed if some files are truncated otherwise the generator will crash
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':

    # Remove old log.txt
    if os.path.exists(pDir_data_dir+'log.txt'):
        os.remove(pDir_data_dir+'log.txt')

    os.makedirs(os.path.dirname(pDir_models), exist_ok=True)

    # Save a copy of the parsetup.ini in the models directory
    #shutil.copy(parsetup_fname,main_directory+'/mypackage/callbacks/'+parsetup_fname)
    shutil.copy(parsetup_fname, pDir_models)

    # Setup parameters from initialization file
    # for changing parameters see parsetup.ini
    # for explanation on parameters see param_README.txt
    config = configparser.ConfigParser()
    try:
        config.read(parsetup_fname)
    except:
        error_exit(57, "No parsetup.ini file. Initialization file needed.")

    conf_general = config['GENERAL']
    seed = int(conf_general['seed'])
    np.random.seed(seed)
    mtype = conf_general['mtype']
    ORIG_INPUT_DATASET = conf_general['path1'].replace("'","")
    folder_pred = conf_general['folder_pred'].replace("'","")

    conf_model = config['modeling']
    modeling = (conf_model['modeling'] == 'True')
    nb_gpu = int(conf_model['nb_gpu'])
    basemodel = conf_model['basemodel'].replace("'","")
    #rescale_ratio = convert_stringratio(conf_model['rescale_ratio'])
    EPOCHS1 = int(conf_model['nepoch'])
    #EPOCHS2 = int(conf_model['nepoch'])
    bsize_per_gpu = int(conf_model['bsize_per_gpu'])
    INIT_LR = float(conf_model['initial_learning_rate'])
    # literal_eval was used before, but as
    # any executive command can be interpreted with literal_eval,
    # it could create security breach
    # so a safer alternative was choosen
    height,width,channels = convert_stringtuple(conf_model['image_dimension']) # (h,w,ch)

    conf_testing = config['testing']
    testing = (conf_testing['testing'] == 'True')

    conf_pred = config['prediction']
    prediction = (conf_pred['prediction'] == 'True')

    # Preprocessing - dimension reduction
    st2b.trace("[INFO] Applying and/or checking preprocessing of images...")
    height,width,channels = Preprocessing(ORIG_INPUT_DATASET,basemodel,height,width,channels).apply_preprocessing()

    # Split raw data folder into train, validation and test folders
    st2b.trace("[INFO] Split raw data folder into train, validation and test folders...")
    check_for_dataset_folders(data_prepro_dir, seed)

    # Define the optimizer at the beginning as needed for modeling and testing
    opt = Adam(lr=INIT_LR, beta_1=0.9)

    # Defining the modules and parameters used accoding to the type of model
    if mtype == 'rgbtm':
        MyClass = ClassManager
        metric_name = metric_name_classification
        metric_name_cbk = metric_name_cbk_classification
        mygenerator = GeneratorModifier
        stractivation = stractivation_classification
        theloss = theloss_rgbtm
        MyTestingBlock = TestingBlock
        MyTestingReporting = TestReporting_multi
        MyModelConverter = ModelConverter
    elif mtype == 'rgbtb':
        MyClass = ClassManagerBinary
        metric_name = metric_name_classification
        metric_name_cbk = metric_name_cbk_classification
        mygenerator = GeneratorModifier
        stractivation = stractivation_classification
        theloss = theloss_rgbtb
        MyTestingBlock = TestingBlock
        MyTestingReporting = TestReporting
        MyModelConverter = ModelConverter
    elif mtype == 'rgbtr':
        MyClass = ClassManagerReg #  nbClasses should be 1
        metric_name = metric_name_regression
        metric_name_cbk = metric_name_cbk_regression
        mygenerator = GeneratorModifier_regression
        stractivation = stractivation_regression
        theloss = theloss_rgbtr
        MyTestingBlock = TestingBlock_regression
        MyTestingReporting = TestReporting_regression
        MyModelConverter = ModelConverter_regression
    else:
        error_exit(56, "Unknown mtype in parsetup.ini. Please check.")

    st2b.trace("[INFO] Determine class weights, number of classes and class mode...")
    classWeight, nbClasses, class_mode_name = MyClass().get_classes()
    st2b.trace("class weights", classWeight)
    st2b.trace("nb of classes", nbClasses)
    st2b.trace("class mode name", class_mode_name)

    # Compute batch size (BS) from the batch sizer per gpu number
    if nb_gpu == 0: nb_gpu = 1
    BS = nb_gpu * bsize_per_gpu

    # Get the corresponding scaling used to pre-trained the model
    scaling = ModelPreprocessInput(basemodel).get_preprocessing_input()

    # initialize the training training data augmentation object
    trainAug = ImageDataGenerator(
        preprocessing_function=scaling,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

    # initialize the validation (and testing) data augmentation object
    valAug = ImageDataGenerator(preprocessing_function=scaling)

    # initialize the training generator
    trainGen = trainAug.flow_from_directory(
        TRAIN_PATH,
        class_mode = class_mode_name,
        target_size=(height, width), # height, width
        color_mode="rgb",
        shuffle=True,
        batch_size=BS)

    # initialize the validation generator
    valGen = valAug.flow_from_directory(
        VAL_PATH,
        class_mode = class_mode_name,
        target_size=(height, width),
        color_mode="rgb",
        shuffle=False,
        batch_size=BS)

    # initialize the testing generator
    testGen = valAug.flow_from_directory(
        TEST_PATH,
        class_mode = class_mode_name,
        target_size=(height, width),
        color_mode="rgb",
        shuffle=False,
        batch_size=BS)

    # Modify classes for regression if regression mode selected
    trainGen = mygenerator(trainGen).correct_label_classes()
    valGen = mygenerator(valGen).correct_label_classes()
    testGen = mygenerator(testGen).correct_label_classes()

    # Compute the total number of samples for train and validation sets
    totalTrain = trainGen.samples
    totalVal = valGen.samples

    # MODELING on TRAIN and VALIDATION SETS AND EVALUATING on TEST SET
    name = 'final'
    # Determine the type of model for saving and reading (warmup or final)
    warmupmc = MyModelConverter(name, trainGen)

    #model.summary()

    # Setup parallel gpu or not deepending on the number of gpu setup
    if nb_gpu <= 1:

        model = CreateModel(basemodel,height,width,channels,nbClasses, stractivation).GetModelStructure()

        gpu_model = model
        activate_multi_gpu = False
    else:

        with tf.device("/cpu:0"):

            model = CreateModel(basemodel,height,width,channels,nbClasses, stractivation).GetModelStructureFinetuningVGG16()

            """
            # load the specific network, ensuring the head FC layer sets are left off and
            # load pre-training on ImageNet weights on the specific model
            baseModel = BaseModel(basemodel,height,width,channels).basemodelselect()

            # initialize the new head of the network, a set of FC layers
            # followed by a softmax classifier
            headModel = FCHeadNet.build(baseModel, nbClasses, stractivation, 256)

            # place the head FC model on top of the base model -- this will
            # become the actual model we will train
            model = Model(inputs=baseModel.input, outputs=headModel)

            # STEP 1 - WARM UP
            # loop over all layers in the base model and freeze them so they
            # will *not* be updated during the training process
            for layer in baseModel.layers:
                layer.trainable = False
            """

        gpu_model = multi_gpu_model(model)
        activate_multi_gpu = True

    #gpu_model.summary()


    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable
    st2b.trace("[INFO] Compiling and training model...")
    # optimizer define just after splitting the data at the beginning
    gpu_model.compile(loss=theloss, optimizer=opt, metrics=[metric_name])  # working!

    # Monitoring functions called after each epoch
    callbacks_list = [
        # Save model after each epoch regardless of its accuracy
        MyCbk(model, top_model_warmup_weights_path), # doesn't work with multi_gpu_model, need _model
        # Rough estimate of time left for the training
        MyCbkTimeEstimation(EPOCHS1),
        EarlyStopping(monitor='val_loss', patience=100, verbose=1),
        ReduceLROnPlateau(monitor='val_'+metric_name_cbk, factor=0.5, patience=5, min_lr=0.000001, verbose=1)
    ]

    # fit the model
    start = time.time()
    H = gpu_model.fit_generator(
        trainGen,
        steps_per_epoch=totalTrain // BS,
        validation_data=valGen,
        validation_steps=totalVal // BS,
        class_weight=classWeight,
        epochs=EPOCHS1,
        callbacks=callbacks_list,
        use_multiprocessing = activate_multi_gpu,
        workers=8
        )
    end = time.time()
    timer(start,end,st2b)

    # Create directory for leanring curves
    os.makedirs(os.path.dirname(pDir_plots_lc), exist_ok=True)

    # Save the model to disk
    st2b.trace("[INFO] Serializing final model...")
    # save the model architecture
    warmupmc.save_model_architecture_json(model)
    del model
    del gpu_model

    # QUICK MODEL EVALUATION ON TEST SET

    st2b.trace("[INFO] Loading and compiling model from disk...")
    loaded_model = warmupmc.load_model_json_w_weights()
    #model = compile_model(mtype, loaded_model, opt, metric_name)
    loaded_model.compile(loss=theloss, optimizer=opt, metrics=[metric_name])

    # reset the testing generator and then use our trained model to
    # make predictions on the data
    # Also generate the Leanring Curves from modeling history
    st2b.trace("[INFO] Evaluating final network on test set..")
    MyTestingBlock(testGen, loaded_model, BS, H, st2b, metric_name, metric_name_cbk,name)

    # save model in tensorflow format
    warmupmc.convert_keras_to_tensorflow(False, loaded_model)

    """
    # STEP 2 - FINE TUNING - OLD NOT UPDATED

    #import keras.backend as K

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and
    # the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch
    # is selected and not the last one.
    model.load_weights(top_model_weights_path)

    # place the head FC model on top of the base model -- this will
    # become the actual model we will train
    #model = Model(inputs=baseModel.input, outputs=headModel)

    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    for layer in baseModel.layers[15:]:
        layer.trainable = True

    # for the changes to the model to take affect we need to recompile
    # the model, this time using SGD with a *very* small learning rate
    st2b.trace("[INFO] re-compiling model...")
    last_lr = float(K.get_value(model.optimizer.lr))
    #opt = SGD(lr=last_lr, momentum=0.9)
    opt = Adam(lr=0.0001, beta_1=0.9)
    #opt = Nadam(lr=last_lr, beta_1=0.9)
    st2b.trace("[INFO] initial learning rate for final model:", INIT_LR)
    st2b.trace("[INFO] previous final learning rate for warm-up model:", last_lr)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    final_model_weights_path = pDir_models+'final_weights.h5'
    callbacks_list = [
        ModelCheckpoint(final_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', min_delta=0.01, patience=15, verbose=1),
        #LearningRateScheduler(step_decay, verbose=1)
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, min_lr=0.000001, verbose=1)
    ]

    # fit the model
    #trainGen.reset()
    #valGen.reset()
    start = time.time()
    H = model.fit_generator(
        trainGen,
        steps_per_epoch=totalTrain // BS,
        validation_data=valGen,
        validation_steps=totalVal // BS,
        class_weight=classWeight,
        epochs=EPOCHS2,
        callbacks=callbacks_list)
    end = time.time()
    timer(start,end,st2b)

    plotAccLoss(H, pDir_plots_lc+"acc_loss_curves_final.png")

    # save the model to disk
    st2b.trace("[INFO] serializing final model...")
    model.save(pDir_models+model_final_h5)

    del model
    """


    if testing:

        # TESTING AND REPORTING - TEST SET

        os.makedirs(os.path.dirname(pDir_results), exist_ok=True)

        st3.trace("[INFO] Loading and compiling final model from disk...")
        with tf.Session() as sess:
            warmupmc = MyModelConverter('final', testGen)
            loaded_model = warmupmc.load_model_json_w_weights()
            loaded_model.compile(loss=theloss, optimizer=opt, metrics=[metric_name])

            # reset the testing generator and then use our trained model to
            # make predictions on the data
            st3.trace("[INFO] Testing final model on test set...")
            MyTestingReporting(testGen, loaded_model, BS, st3)
