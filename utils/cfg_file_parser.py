import configparser as cp

def parser_cfg_file(cfg_file):
    net_params = {}
    train_params = {}
    net_structure_params = []

    config = cp.ConfigParser()
    config.read(cfg_file)

    for section in config.sections():
        # 获取配置文件中的net信息
        if section == 'net':
            for option in config.options(section):
                net_params[option] = config.get(section,option)

        # 获取配置文件中的train信息
        if section == 'train':
            for option in config.options(section):
                train_params[option] = config.get(section,option)

        # 获取配置文件中的convolution配置信息
        if 'convolution' in section:
            conv_params = {}
            for option in config.options(section):
                conv_params[option] = config.get(section,option)
            net_structure_params.append(conv_params)

        # 获取配置文件中的max_pool配置信息
        if 'maxpool' in section:
            maxpool_params = {}
            for option in config.options(section):
                maxpool_params[option] = config.get(section,option)
            net_structure_params.append(maxpool_params)

        # 获取配置文件中的connected配置信息
        if 'connected' in section:
            connected_params = {}
            for option in config.options(section):
                connected_params[option] = config.get(section,option)
            net_structure_params.append(connected_params)

    return net_params,train_params,net_structure_params

if __name__=='__main__':
    net_params, train_params, net_structure_params = parser_cfg_file('../cfg/tiny-yolo.cfg')
    # print(net_params)
    # print(train_params)
    print(net_structure_params)




