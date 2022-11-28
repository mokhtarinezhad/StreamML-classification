


import sys
import os
import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, \
RobustScaler, Normalizer, QuantileTransformer

image_extensions = ['.bmp','.dib','.jpeg','.jpg','.JPG','.JPEG','.jpe','.jp2',\
                    '.png','.PNG','.webp','.pbm','.pgm','.ppm','.sr', '.ras',\
                    '.tiff', '.tif','.TIFF', '.TIF']

# define the path separator mark
sepfn = '/'

# define the default file name and path
model_js = 'model.json'
#saved_models = 'models'
weights_h5 = 'weights.h5'

model_warmup_weights_h5 = 'warmup_weights.h5'
model_final_weights_h5 = 'final_weights.h5'

model_warmup_h5 = 'warmup_model.h5'
model_final_h5 = 'final_model.h5'

model_warmup_json = 'warmup_model.json'
model_final_json = 'final_model.json'

parsetup_fname = "parsetup.ini"

labels_pickle_fname = "labels.p"
encoder_pickle_fname = "encoder.p"
scaler_pickle_fname = "scaler.p"
weights_prepro_pickle_fname = "weights_prepro.p"

main_directory = os.getcwd() # where the python code is lunched
#main_directory = '/notebooks/code'
#main_directory = sepfn+"notebooks"+sepfn+"RGBtrlearning_git"

ext = '.png' # extension to save the plots
plots_dir = 'plots'
models_dir = 'models'
results_dir = 'results'
predictions_dir = 'predictions'
data_dir = 'data'
data_prepro_dir = data_dir+sepfn+'preprocessed'
pDir_data_dir = main_directory + sepfn + data_dir + sepfn
pDir_plots = main_directory + sepfn + data_dir + sepfn + plots_dir + sepfn
pDir_models = main_directory + sepfn + data_dir + sepfn+models_dir + sepfn
pDir_results = main_directory+sepfn+ data_dir+sepfn+results_dir+sepfn
pDir_predictions = main_directory+sepfn+data_dir+sepfn+ predictions_dir+sepfn
pDir_plots_lc = pDir_models+'learningcurves'+sepfn
pDir_data_prepro = main_directory+sepfn+data_prepro_dir+sepfn

top_model_warmup_weights_path = pDir_models+model_warmup_weights_h5
top_model_final_weights_path = pDir_models+model_final_weights_h5

# tensorflow serving compatible - need folder '1' as default to start
#tfmodel_dir0 = 'model'+sepfn
tfmodel_dir0_warmup = 'model'+sepfn # previously: 'warmup'
tfmodel_dir0_model = 'model'+sepfn
tfmodel_dir_ver = '1'+sepfn
#tfmodel_dir = tfmodel_dir0+tfmodel_dir_ver

# Data splitting parameters

# initialize the path to the *original* input directory of images
#ORIG_INPUT_DATASET = main_directory+sepfn+data_dir+sepfn+"raw"
#ORIG_INPUT_DATASET = '/notebooks/RGBtrlearning_binary/original/datasets/PetImages'
# setup in parsetup.ini

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = main_directory+sepfn+data_dir+sepfn+"idc"

train_name = "training"
val_name = "validation"
test_name = "testing"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, train_name])
VAL_PATH = os.path.sep.join([BASE_PATH, val_name])
TEST_PATH = os.path.sep.join([BASE_PATH, test_name])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1

path_to_weights = main_directory+sepfn+'mypackage'+sepfn+'nn'+sepfn
vgg16_weights_notop = path_to_weights + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
inceptionv3_weights_notop = path_to_weights + 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50_weights_notop = path_to_weights + 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
xception_weights_notop = path_to_weights + 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
mobilenet_weights_notop = path_to_weights + 'mobilenet_1_0_128_tf_no_top.h5'
inceptionresnetv2_weights_notop = path_to_weights + 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_weights_notop = path_to_weights + 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
nasnet_weights_notop = path_to_weights + 'NASNet-large-no-top.h5'
efn_weights_notop = path_to_weights + 'adv.prop.notop-b7.h5'
# Scaling parameters
scaling_dict =	{
  "minmaxscaler": MinMaxScaler(feature_range=(0, 1)),
  "standardscaler": StandardScaler(),
  "maxabsscaler": MaxAbsScaler(),
  "robustscaler": RobustScaler(quantile_range=(25, 75)),
  "quantiletransformer_normal": QuantileTransformer(output_distribution='normal'),
  "quantiletransformer_uniform": QuantileTransformer(output_distribution='uniform'),
  "Normalizer": Normalizer(),
  "none": None
}


metric_name_classification = "accuracy"
metric_name_cbk_classification = "acc"
stractivation_classification  = "softmax"
theloss_rgbtm = "categorical_crossentropy"
theloss_rgbtb = "binary_crossentropy"

metric_name_regression = "mean_absolute_error"
metric_name_cbk_regression = metric_name_regression
stractivation_regression = "linear"
theloss_rgbtr = "mean_squared_error"


# Manage traces
import trace
import writing

swdt0 = writing.SimpleWriter(pDir_data_dir+'log.txt', sep = ' ', mode = 'w')
dt0 = writing.doubleTracer(writing.tty, swdt0)
st0 = trace.Tracing(dt0, trace.ListTracing, run=True)

swdt = writing.SimpleWriter(pDir_models+'output.txt', sep = ' ', mode = 'w')
dt = writing.doubleTracer(writing.tty, swdt)
st2 = trace.Tracing(dt, trace.ListTracing, run=True)

swdt1 = writing.SimpleWriter(pDir_models+'output_fmodel.txt', sep = ' ', mode = 'w')
dt1 = writing.doubleTracer(writing.tty, swdt1)
st2b = trace.Tracing(dt1, trace.ListTracing, run=True)

swdt2 = writing.SimpleWriter(pDir_results+'output_test.txt', sep = ' ', mode = 'w')
dt2 = writing.doubleTracer(writing.tty, swdt2)
st3 = trace.Tracing(dt2, trace.ListTracing, run=True)

swdt3 = writing.SimpleWriter(pDir_predictions+'output_pred.txt', sep = ' ', mode = 'w')
dt3 = writing.doubleTracer(writing.tty, swdt3)
st4 = trace.Tracing(dt3, trace.ListTracing, run=True)

preprocessing_file_path = os.path.join(data_prepro_dir, 'preprocessing.txt')
swdt4 = writing.SimpleWriter(preprocessing_file_path, sep = ' ', mode = 'w')
dt4 = writing.doubleTracer(writing.tty, swdt4)
st5 = trace.Tracing(dt4, trace.ListTracing, run=True)

def convert_stringtuple(stringtuple):
    stringtuple = stringtuple[1:-1] # remove the brackets
    h,w,ch = stringtuple.split(',') # heigth, width, channel
    h = h.strip()
    w = w.strip()
    ch = ch.strip()
    #return int(h),int(w),int(ch)
    return h,w,ch

def error_exit(n_exit, *messageParam):
    """ Print a fatal error message with a list of parameters
    identifying the error.
    @param n_exit int: the sys.exit parameter
    @param messageParam list: printable messages
    """
    st0.trace('Fatal error:', *messageParam)
    sys.exit(n_exit)

def error_except(n_exit, *messageParam):
    """ Print a fatal error message while an exception is raised
    with a list of parameters identifying the error.
    @param n_exit int: the sys.exit parameter
    @param messageParam list: printable messages
    """
    st0.trace('Fatal error:', *messageParam)
    print(sys.exc_info())
    sys.exit(n_exit)

def pickleSave(fname, theData):
    """ Error and exception handeling for pickle.dump (save)
    @param fname str: file name of the pickle file to save
    @param theData obj: Python object structure to pickle
    """
    with open(fname, 'wb') as f:
        try:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(theData, f, pickle.HIGHEST_PROTOCOL)
        except:
            error_except(12, 'pickle write KO', fname)
    f.close()

def pickleRead(fname):
    """ Error and exception handeling for pickle.load
    @param fname str: file name of the pickle file to save
    @return: Python object structure unpickled
    """
    with open(fname, 'rb') as f:
        try:
            # Pickle the 'data' dictionary using the highest protocol available.
            theData = pickle.load(f)
        except:
            error_except(12, 'pickle read KO', fname)
    f.close()
    return theData

def folderexists(fname):
    """ Error and exception handeling for pickle.load
    @param fname str: ffolder name
    @return: bool True folder exists
    """
    doExist = os.path.exists(fname)
    if doExist == False:
        error_exit(91, 'No folder named', fname)
    else:
        return True

def get_last_model_version(path):
    """ Return the latest model version in the model directory
    @param path str: path of the directory where the version folders are stored
    @return int: latest model version number
    """
    dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]

    dirlist_tmp = dirlist.copy()
    for cnt,element in enumerate(dirlist):
        if not element.isdigit():
            dirlist_tmp.remove(element)
    dirlist_tmp = np.array(dirlist_tmp)
    dirlist_tmp = dirlist_tmp.astype(np.int)

    dirlist_tmp.sort()
    last_version_nb = dirlist_tmp[-1]
    return last_version_nb

def last_dirs_model(default_path, tfmodel_dir0):
    """ Return latest model path and if the model directory exists or not
    @param default_path str: default path of the model directory
    @return default_path_exist bool: if the  default path of the model directory exists or not
    @return current_tfmodel_dir str: partial path of the latest model inside model directory
    """
    if os.path.isdir(default_path):
        last_version_nb = get_last_model_version(default_path)
        current_tfmodel_dir = tfmodel_dir0+str(last_version_nb)+sepfn
        default_path_exist = True
    else:
        current_tfmodel_dir = tfmodel_dir0+tfmodel_dir_ver
        default_path_exist = False
    return default_path_exist, current_tfmodel_dir

def version_dirs_model(version,tfmodel_dir0):
    """ Return the selected model version path from the model directory
    @param path str: path of the directory where the version folders are stored
    @return int: selected model version number
    """
    version_tfmodel_dir = tfmodel_dir0+version+sepfn
    if os.path.isdir(version_tfmodel_dir):
        return version_tfmodel_dir
    else:
        error_exit(91, 'No version', version)

def next_dirs_model(default_path, tfmodel_dir0):
    """ Return latest model path and if the model directory exists or not
    @param default_path str: default path of the model directory
    @return default_path_exist bool: if the  default path of the model directory exists or not
    @return next_tfmodel_dir str: partial path of the next model inside model directory
    """
    if os.path.isdir(default_path):
        last_version_nb = get_last_model_version(default_path)
        next_tfmodel_dir = tfmodel_dir0+str(last_version_nb+1)+sepfn
        default_path_exist = True
    else:
        next_tfmodel_dir = tfmodel_dir0+tfmodel_dir_ver
        default_path_exist = False
    return default_path_exist, next_tfmodel_dir
