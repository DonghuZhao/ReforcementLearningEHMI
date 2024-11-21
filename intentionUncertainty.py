from dynamicBayesianNetworkModel import *
from data_preprocess import *
from config import *

# HyperParams
INTENTION_TRANS_PROB = 0.1

def main(mode_):
    """main function"""

    # 训练数据格式转换
    train_data1, test_data1 = trainDataTake()
    # train_data2, test_data2 = trainDataTake2()

    dbn = dynamicBayesianNetworkModel(file_path=MODEL_PATH)

    # 训练模型
    if mode_['train']:
        dbn.train(train_data1, save_model=mode_['save'])
    else:
        dbn.load_model()

    # 评估模型
    if mode_['evaluate']:
        dbn.evaluate(test_data1)


if __name__ == "__main__":
    # 设置NUMEXPR_MAX_THREADS环境变量
    os.environ["NUMEXPR_MAX_THREADS"] = "8"  # 设置为所需的线程数
    pd.options.mode.chained_assignment = None  # 默认为 'warn'

    # 设置训练和评估的类型
    # [是否训练、是否保存模型、是否评估模型]
    mode_select = {
        'notSave': {'train': True, 'save': False, 'evaluate': True},
        'evaluateOnly': {'train': False, 'save': False, 'evaluate': True},
        'updateModel': {'train': True, 'save': True, 'evaluate': True}
    }

    mode = 'evaluateOnly'
    mode_ = mode_select[mode]

    main(mode_)


