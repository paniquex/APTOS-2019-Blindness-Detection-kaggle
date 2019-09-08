


from keras.models import model_from_json
import cv2
import numpy as np


class VesselNet():

    def __init__(self, path):
        self.path = path
        self.load_model()

    def load_model(self):
        self.model = model_from_json(open(self.path + "VesselNet" + '_architecture.json').read())
        self.model.load_weights(self.path + "VesselNet" + '_best_weights.h5')

    def predict(self, image):
        orgImg_temp = image
        orgImg=orgImg_temp[:,:,1] * 0.75 + orgImg_temp[:,:,0] * 0.25
        
        height,width=orgImg.shape[:2]
        orgImg = np.reshape(orgImg, (height,width,1))
        patches_pred, new_height, new_width, adjustImg = get_test_patches(orgImg)

        predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
        pred_patches = pred_to_patches(predictions)

        pred_imgs = recompone_overlap(pred_patches, new_height,new_width)
        pred_imgs=pred_imgs[:,0:height,0:width,:]

        adjustImg = adjustImg[0,0:height,0:width,:]
        print(adjustImg.shape)
        probResult = pred_imgs[0,:,:,0]

        return (probResult*255).astype(np.uint8)

#         binaryResult = gray2binary(probResult)
#         resultMerge = visualize([adjustImg,binaryResult],[1,2])
#         resultMerge = cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)
        
        

#######################################################################################################

def img_process(data):
    assert(len(data.shape)==4)
    data=data.transpose(0, 3, 1,2)
    train_imgs=np.zeros(data.shape)
    for index in range(data.shape[1]):
        train_img=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
        train_img[:,0,:,:]=data[:,index,:,:]
        #print("original",np.max(train_img),np.min(train_img))
        train_img = dataset_normalized(train_img)   #归一化
        #print("normal",np.max(train_img), np.min(train_img))
        train_img = clahe_equalized(train_img)      #限制性直方图归一化
        #print("clahe",np.max(train_img), np.min(train_img))
        train_img = adjust_gamma(train_img, 1.2)    #gamma校正
        #print("gamma",np.max(train_img), np.min(train_img))
        train_img = train_img/255.  #reduce to 0-1 range
        #print("reduce",np.max(train_img), np.min(train_img))
        train_imgs[:,index,:,:]=train_img[:,0,:,:]
    train_imgs=train_imgs.transpose(0, 2, 3, 1)
    
    return train_imgs

def paint_border(imgs):
    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]  # height of the full image
    img_w = imgs.shape[2]  # width of the full image
    leftover_h = (img_h - 96) % 5  # leftover on the h dim
    leftover_w = (img_w - 96) % 5  # leftover on the w dim
    full_imgs=None
    if (leftover_h != 0):  #change dimension of img_h
        tmp_imgs = np.zeros((imgs.shape[0],img_h+(5-leftover_h),img_w,imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:img_h,0:img_w,0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_w+(5 - leftover_w),full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:imgs.shape[1],0:img_w,0:full_imgs.shape[3]] =imgs
        full_imgs = tmp_imgs
        
    return full_imgs

def extract_patches(full_imgs):
    assert (len(full_imgs.shape)==4)  #4D arrays
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image

    assert ((img_h-96)%5==0 and (img_w-96)%5==0)
    N_patches_img = ((img_h-96)//5+1)*((img_w-96)//5+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    patches = np.empty((N_patches_tot,96,96,full_imgs.shape[3]))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-96)//5+1):
            for w in range((img_w-96)//5+1):
                patch = full_imgs[i,h*5:(h*5)+96,w*5:(w*5)+96,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    
    return patches

def get_test_patches(img):
    test_img = []
    test_img.append(img)
    test_img = np.asarray(test_img)
    test_img_adjust = img_process(test_img)
    test_imgs = paint_border(test_img_adjust)
    test_img_patch = extract_patches(test_imgs)

    return test_img_patch, test_imgs.shape[1], test_imgs.shape[2], test_img_adjust

def pred_to_patches(pred):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0],pred.shape[1],1+1))  #(Npatches,height*width)
    pred_images[:,:,0:1+1]=pred[:,:,0:1+1]
    pred_images = np.reshape(pred_images,(pred_images.shape[0],96,96,1+1))
    
    return pred_images

def recompone_overlap(preds,img_h,img_w):
    assert (len(preds.shape)==4)  #4D arrays

    patch_h = 96
    patch_w = 96
    N_patches_h = (img_h-patch_h)//5+1
    N_patches_w = (img_w-patch_w)//5+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//5+1):
            for w in range((img_w-patch_w)//5+1):
                full_prob[i,h*5:(h*5)+patch_h,w*5:(w*5)+patch_w,:]+=preds[k]
                full_sum[i,h*5:(h*5)+patch_h,w*5:(w*5)+patch_w,:]+=1
                k+=1
    print(k,preds.shape[0])
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print('using avg')
    return final_avg

def gray2binary(image,threshold=0.5):
    image = (image >= threshold) * 1
    
    return image

def visualize(image,subplot):
    row=int(subplot[0])
    col=int(subplot[1])
    height,width=image[0].shape[:2]
    result=np.zeros((height*row,width*col,3))

    total_image=len(image)
    index=0
    for i in range(row):
        for j in range(col):
            row_index=i*height
            col_index=j*width
            if index<total_image:
                try:
                    result[row_index:row_index+height,col_index:col_index+width,:]=image[index]*255
                except:
                    result[row_index:row_index + height, col_index:col_index + width, 0] = image[index]*255
                    result[row_index:row_index + height, col_index:col_index + width, 1] = image[index]*255
                    result[row_index:row_index + height, col_index:col_index + width, 2] = image[index]*255
            index=index+1
    result=result.astype(np.uint8)
    return result

#######################################################################################################

def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

#######################################################################################################


def normal_crop(image, IMAGE_SIZE):
    # Compute the center (cx, cy) and radius of the eye
    cy = image.shape[0]//2
    midline = image[cy,:]
    midline = np.where(midline > midline.mean()/3)[0]
    if len(midline) > image.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = image.shape[1]//10, 9*image.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    # Crops, resizes the image to Image
    scaling = IMAGE_SIZE/(2 * r)
    rotation = 0
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - IMAGE_SIZE/2
    M[1,2] -= cy - IMAGE_SIZE/2

    image = cv2.warpAffine(image, M, ( IMAGE_SIZE, IMAGE_SIZE ))

    return image

def circle_crop(img, IMAGE_SIZE):
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = normal_crop(img, IMAGE_SIZE)

    return img

def clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return clahed
def ben(image, sigmaX=30):
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)

        return img

def extract_bv(image):
    (b, green_fundus, r) = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area

    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 10000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
#     blood_vessels = cv2.bitwise_not(finimage)
    return finimage

def Krish(crop):    
    Input=crop[:,:,1]    
    a,b=Input.shape    
    Kernel=np.zeros((3,3,8))#windows declearations(8 windows)    
    Kernel[:,:,0]=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])     
    Kernel[:,:,1]=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])    
    Kernel[:,:,2]=np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])    
    Kernel[:,:,3]=np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])    
    Kernel[:,:,4]=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])    
    Kernel[:,:,5]=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])    
    Kernel[:,:,6]=np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])    
    Kernel[:,:,7]=np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])    
    #Kernel=(1/float(15))*Kernel    
    #Convolution output    
    dst=np.zeros((a,b,8))    
    for x in range(0,8):        
        dst[:,:,x] = cv2.filter2D(Input,-1,Kernel[:,:,x])    
    Out=np.zeros((a,b))    
    for y in range(0,a-1):        
        for z in range(0,b-1):            
            Out[y,z]=max(dst[y,z,:])    
    Out=np.uint8(Out)
    # print(Out)
    return Out





