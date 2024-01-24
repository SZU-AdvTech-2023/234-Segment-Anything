# This file is used to configure the training parameters for each task

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "/data/gjx/project/dataset/CAMUS_256_video/tvt"  #
    data_subpath = "../../dataset/SAMUS/Echocardiography-CAMUS/" 
    save_path = "./checkpoints/CAMUS/SAMUS_video/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    # load_path = save_path + "/SAMUS_11171158_0_0.6666666666984046.pth"
    load_path = "/data/gjx/project/SAMUS/SAMUS/checkpoints/CAMUS/SAMUS_video/SAMUS_12.14-17:53_3.pth"
    # 23/30: dice: 91.59 HD: 13.32
    save_path_code = "_"
    visual_result_path = "/data/gjx/project/SAMUS/SAMUS/visual_result"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 30                       # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                       # thenumber of classes (background + foreground)
    img_size = 256                      # the input size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "SAMUS"

# ==================================================================================================
def get_config(task="US30K"):
    if task == "CAMUS":
        return Config_CAMUS()
    else:
        assert("We do not have the related dataset, please choose another task.")