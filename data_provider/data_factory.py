from data_provider.data_loader import  Dataset_stock36index, \
    Dataset_stock2geo, Dataset_stock3geo,Dataset_stock,Dataset_stock45index,Dataset_custom_stock,Dataset_stock0index#, Dataset_custom_groundwater, Dataset_groundwater
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {

    # 'custom': Dataset_Custom,
    # 'custom_stock':Dataset_custom_stock,
    'stock':Dataset_stock,
    'stock0index':Dataset_stock0index,
    'stock2geo':Dataset_stock2geo,
    'stock36index':Dataset_stock36index,
    'stock45index':Dataset_stock45index,
    'stock3geo':Dataset_stock3geo,
    # 'custom_groundwater':Dataset_custom_groundwater,
    # 'groundwater':Dataset_groundwater,

}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        # if args.data =='stock' or args.data =='custom_stock'  or args.data =='custom' or 'stock' in args.data or args.data =='stock2geo':
        data_path=['train.pkl','val.pkl','test.pkl']
        # if args.data =='groundwater' or args.data =='custom_groundwater':
        #     data_path=['train.pkl','val.pkl','test.pkl']
        # else:
        #     data_path=['train.csv','val.csv','test.csv']
        data_set = Data(
            root_path=args.root_path,
            data_path=  data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            index_num=args.index_num,
            stocknums=args.stocknums
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader
    