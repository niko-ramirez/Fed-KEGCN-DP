import tensorflow as tf
import random
import copy

class Server:
    def __init__(self, globalModel, device, logging, writer = None) -> None:
        # log setting
        self.logging = logging
        self.writer = writer

        # server setting
        self.device = device
        self.Epoch = -1                        # record the FL round by self
        self.round = random.randint(0, 1e8)    # record the mask round of FL this round
        self.local_state_dict = []
        self.val_acc = 0
        self.model = globalModel
        self.global_state_dict = copy.deepcopy(globalModel.layers)

        # loss function
        self.loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2()

    def aggregate(self):
        pass
        # clientNum = len(self.local_state_dict)
        # if clientNum == 0:
        #     return 
        # self.val_acc /= clientNum
        # self.Epoch += 1
        
        # self.logging.info("Clients Val Avg Acc: {:>8f}".format(self.val_acc))
        # # self.writer.add_scalar(f"val/acc/avg", self.val_acc, self.Epoch)

        # # aggregate all parameter        
        # for layer_name in self.local_state_dict[0].keys():
        #     self.global_state_dict[layer_name] = tf.zeros_like(self.global_state_dict[layer_name])
            
        #     for localParame in self.local_state_dict:
        #         self.global_state_dict[layer_name].add_(localParame[layer_name])
            
        #     self.global_state_dict[layer_name].div_(clientNum)

        # self.local_state_dict.clear()
        # self.val_acc = 0
        # # self.model.load_state_dict(self.global_state_dict)
        # self.round = random.randint(0, 1e8) 

    def sendParame(self):
        return self.round, self.global_state_dict

    def getParame(self, round, localParame, val_acc):
        if round == self.round:
            self.local_state_dict.append(localParame)
            self.val_acc += val_acc