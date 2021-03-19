%matplotlib inline
import tifffile as tiff
from matplotlib import pyplot as plt 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import math

def normalize(img):
    min = img.min()
    max = img.max()
    return 2.0 * (img - min) / (max - min) - 1.0


imagefiles_id = ['02','03','06','08','10','13',
              '14','15','16','18','19','20','21','01','04','05','07','09','11','12','17','22','23','24']
train_img_normalized=[normalize(tiff.imread('data/mband/{}.tif'.format(i))) for i in imagefiles_id]


if __name__ == '__main__':
    
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in imagefiles_id:
        img_m = normalize(tiff.imread('data/mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
        mask = tiff.imread('data/gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')
    

# define U-Net structure 
def unet_model(n_classes=5, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.3, 0.1, 0.1, 0.3]):
    
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    bathnorm=BatchNormalization()(conv2)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bathnorm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    dropout=Dropout(0.2)(conv4)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(dropout)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

    n_filters //= growth_factor
    if upconv:
        up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6)
    bathnorm1=BatchNormalization()(conv6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(bathnorm1)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    dropout1=Dropout(0.3)(conv8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(dropout1)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=SGD(), loss=weighted_binary_crossentropy)
    return model


if __name__ == '__main__':
    model = unet_model()
    print(model.summary())
    plot_model(model, to_file='unet_model.png', show_shapes=True)
    
    
    
# help function -- make overlapping prediction 


def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]

    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height/patch_sz) 
    npatches_horizontal = math.ceil(img_width/patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirror reflections of neighbors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2*img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2*img_width - j - 1, :]

    # now assemble all patches in one array
    patches_list = []
    for i in np.arange(0, npatches_vertical-0.5,0.5):
        for j in np.arange(0, npatches_horizontal-0.5,0.5):
            x0, x1 = int(i * patch_sz), int((i + 1) * patch_sz)
            y0, y1 = int(j * patch_sz), int((j + 1) * patch_sz)
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=32) 
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)


    k=0
    for i in np.arange(0, npatches_vertical-0.5,0.5):
        for j in np.arange(0, npatches_horizontal-0.5,0.5):
            if j ==0 and i == 0:
                x0, x1 = int(i * patch_sz), int((i + 1) * patch_sz)
                y0, y1 = int(j * patch_sz), int((j + 1) * patch_sz)
                prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :] 
                k +=1
            elif j == 0 and i !=0:
                x0, x1 = int(i * patch_sz), int((i + 1) * patch_sz)
                y0, y1 = int(j * patch_sz), int((j + 1) * patch_sz)
                prediction[(x0+40):x1, y0:y1, :] = patches_predict[k, 40:, :, :] 
                k +=1
            elif j != 0 and i !=0:
                x0, x1 = int(i * patch_sz), int((i + 1) * patch_sz)
                y0, y1 = int(j * patch_sz), int((j + 1) * patch_sz)
                prediction[(x0+40):x1, (y0+40):y1, :] = patches_predict[k, 40:, 40:, :] 
                k +=1
            else:
                x0, x1 = int(i * patch_sz), int((i + 1) * patch_sz)
                y0, y1 = int(j * patch_sz), int((j + 1) * patch_sz)
                prediction[x0:x1, (y0+40):y1, :] = patches_predict[k, :,40:, :]
                k +=1
    return prediction[:img_height, :img_width, :]

def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Buildings (grey)
        1: [223, 194, 125],  # Roads & Tracks (light orange)
        2: [27, 120, 55],    # Trees (green)
        3: [166, 219, 160],  # Crops (greyish-green)
        4: [116, 173, 209]   # Water (blue)
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict


# rotate image before get patches 

import random

def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] > sz \
    and img.shape[1] > sz \
    and img.shape[0:2] == mask.shape[0:2]
    
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    rotate1=random.choice(item)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)][:,::rotate1,:]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)][:,::rotate1,:]
    return patch_img, patch_mask


def get_patches(img, mask, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    return np.array(x), np.array(y)  # keras needs numpy arrays rather than lists


TRAIN_SZ = 200  # train size
x_train, y_train = get_patches(img_train[0], mask_train[0], n_patches=TRAIN_SZ, sz=160)
VAL_SZ = 50     # validation size
x_val, y_val = get_patches(img_validation[0], mask_validation[0], n_patches=VAL_SZ, sz=160) 


def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]
    return patch_img, patch_mask


def get_patches(x_dict, y_dict, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)

## below is training function using all data as one df for training  
def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=4000, sz=160)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=1000, sz=160)
        model = unet_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model
    
# or you can split
x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=4000, sz=160)
x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=1000, sz=160)


## start training 
model_hist=model.fit(x_train, y_train, batch_size=32, epochs=10,
          verbose=True,
          validation_data=(x_val, y_val))
