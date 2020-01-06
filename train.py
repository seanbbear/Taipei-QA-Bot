from preprocess_data import convert_data_to_feature
from core import makeDataset
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW, WarmupLinearSchedule
import torch

from argparse import Namespace

# https://www.cnblogs.com/darkknightzh/p/6591923.html 設定GPU來源
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def accuracy(predict,target):
    _,pred_indices = predict.max(dim=1)
    n_correct = torch.eq(pred_indices,target).sum().item()
    return n_correct / len(pred_indices)





if __name__ == "__main__":
    # # Namespace from argsparse用法: https://github.com/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_3/3_5_Classifying_Yelp_Review_Sentiment.ipynb
    # args = Namespace(
    #     local_rank=2,
    #     n_gpu = 2,
    #     no_cuda = 'store_true',
    # )

    
    # # --------------------------------------------------------------------------------

    # if args.local_rank == -1 or args.no_cuda:
    #         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #         args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl')
    #     args.n_gpu = 1
    # args.device = device
    # print(args.device)

    device = torch.device("cuda")

    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)

    data_feature = convert_data_to_feature()
    input_ids = data_feature['input_ids']
    input_masks = data_feature['input_masks']
    input_segment_ids = data_feature['input_segment_ids']
    answer_lables = data_feature['answer_lables']

    train_dataset, valid_dataset = makeDataset(input_ids = input_ids, input_masks = input_masks, input_segment_ids = input_segment_ids, answer_lables = answer_lables)
    train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=8,shuffle=True)

    config = bert_config.from_pretrained('bert-base-chinese',num_labels = 149)
    model = bert_class.from_pretrained('bert-base-chinese', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.to(device)

    # # 多張GPU使用
    # if args.n_gpu > 1:
    #         model = torch.nn.DataParallel(model)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
    # eps用來加在分母以提高模型穩定度

    epoch_num = 15
    t_total = len(train_dataloader) * epoch_num # 30為epoch數
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)

    
    model.zero_grad()
    for epoch in range(epoch_num):
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(train_dataloader):
            model.train()

            batch_dict = tuple(t.to(device)for t in batch_dict)
            # print(batch_dict[0].shape)
            # print(batch_dict[3].shape)

            outputs = model(
                batch_dict[0],
                # attention_mask=batch_dict[1],
                labels = batch_dict[3]
                )
            loss,logits = outputs[:2]

            # if args.n_gpu > 1:
            #     loss = loss.mean() # 多GPU，loss需取平均
            
            loss.sum().backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            # loss計算
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # accuracy計算
            acc_t = accuracy(logits, batch_dict[3])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            if (batch_index % 10 == 9):
                print("epoch:%2d batch:%4d train_loss: %1.4f train_acc: %1.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
        running_loss_val = 0.0
        running_acc = 0.0
        # validate
        for batch_index, batch_dict in enumerate(valid_dataloader):
            model.eval()

            batch_dict = tuple(t.to(device)for t in batch_dict)
            # print(batch_dict[0].shape)
            # print(batch_dict[3].shape)

            with torch.no_grad():
                outputs = model(
                    batch_dict[0],
                    # attention_mask=batch_dict[1],
                    labels = batch_dict[3]
                    )
                loss,logits = outputs[:2]
            # loss計算
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # accuracy計算
            acc_t = accuracy(logits, batch_dict[3])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            if (batch_index % 10 == 9):
                print("epoch:%2d batch:%4d valid_loss: %1.4f valid_acc: %1.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
        running_loss_val = 0.0
        running_acc = 0.0
    model.save_pretrained('model')
            
            # if args.n_gpu > 1:
            #     loss = loss.mean() # 多GPU，loss需取平均
            
            # loss.sum().backward()
            # optimizer.step()
            # # scheduler.step()  # Update learning rate schedule
            # model.zero_grad()
            
            # print(loss)


