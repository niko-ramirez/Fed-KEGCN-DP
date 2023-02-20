# from dgl._deprecate.graph import DGLGraph
import tensorflow as tf
# import torch
# from torch import nn, optim
from tqdm import tqdm
# from dgl import DGLGraph
import copy
from metrics import *
from utils import *


class Client:
    def __init__(self, client_id, data, model, FLAGS, epoch, lr, l2norm, model_path, logging, eval_func) -> None:
        # client setting
        self.client_id = client_id
        self.model = model
        self.epoch = epoch
        self.flags = FLAGS
        self.logging = logging
        self.get_eval = eval_func
        self.train_data = data["train"]
        self.test_data = data["test"]
        self.valid = data["valid"]
        self.y = data["y"]
        self.support = data["support"]


        self.Epoch = -1                    # record the FL round by self
        self.round = None                  # record the mask round of FL this round from the server
        self.val_acc = None
        self.model_param = None
        self.placeholders = model.placeholders

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        acc_best = 0.
        test_acc = 0.
        self.Epoch += 1

        # Train model
        for epoch in range(self.flags.epochs):
            # Construct feed dictionary
            feed_dict = construct_feed_dict(1.0, self.support, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.flags.dropout})
            # Training step
            outputs = sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
            # Print results
            if epoch % 10 == 0:
                self.logging.info("Epoch: {} train_loss= {:.5f}".format(epoch+1, outputs[1]))

            if epoch % 10 == 0 and self.valid is not None:
                # model.evaluate()
                output_embeddings = sess.run(self.model.outputs, feed_dict=feed_dict)
                train_acc, _ = self.get_eval(output_embeddings[0], self.train_data, self.y, self.logging)
                self.logging.info("Train Accuracy: %.3f" % (train_acc * 100))
                acc, _ = self.get_eval(output_embeddings[0], self.valid, self.y, self.logging)
                self.logging.info("Valid Accuracy: %.3f" % (acc * 100))
                if acc > acc_best:
                    acc_best = acc
                    test_acc, result = self.get_eval(output_embeddings[0], self.test_data, self.y, self.logging)
                self.logging.info("Test Accuracy: %.3f" % (test_acc * 100))


            if epoch % 10 == 0 and epoch > 0 and self.valid is None:
                # model.evaluate()
                output_embeddings = sess.run(self.model.outputs, feed_dict=feed_dict)
                train_acc, _ = self.get_eval(output_embeddings[0], self.train_data, self.y, self.logging)
                self.logging.info("Train Accuracy: %.3f" % (train_acc * 100))
                acc, temp = self.get_eval(output_embeddings[0], self.test_data, self.y, self.logging)
                self.logging.info("Test Accuracy: %.3f" % (acc * 100))
                if acc > acc_best:
                    acc_best = acc
                    result = temp

        self.logging.info("Optimization Finished! Best Valid Acc: {} Test: {}".format(
                        round(acc * 100,2), " ".join([str(round(i*100,2)) for i in result])))
    

    def getParame(self, round, param):
        self.round = round
        if self.model_param is not None:
            for layer in self.model_param:
                if ("layers.0" in layer) or layer.endswith("w_comp"): 
                    param[layer] = copy.deepcopy(self.model_param[layer])
        self.model.load_state_dict(param)


    # # upload the local model's parameters to parameter server
    def uploadParame(self):
        param = {}
        for layer in self.model.layers:
            param[layer] = self.model_param[layer]
        return self.round, param, self.val_acc





    # def __init__(self, client_id, data, model, device, epoch, lr, l2norm, model_path, logging, writer) -> None:
    #     # log setting
    #     self.model_path = model_path
    #     self.logging = logging
    #     self.writer = writer

    #     # client setting
    #     self.client_id = client_id
    #     self.device = device
    #     self.data = data
    #     self.model = model.to(self.device)
    #     self.epoch = epoch

    #     self.Epoch = -1                    # record the FL round by self
    #     self.round = None                  # record the mask round of FL this round from the server
    #     self.val_acc = None
    #     self.model_param = None
        
    #     # training setting
    #     self.loss_fn = nn.CrossEntropyLoss()
    #     self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2norm)

    #     # data setting
    #     # node data process
    #     self.labels = tf.convert_to_tensor(data.labels).view(-1).to(self.device)
    #     # self.labels = torch.from_numpy(data.labels).view(-1).to(self.device)
    #     train_idx = data.train_idx
    #     self.val_idx = train_idx[ : len(train_idx) // 5]
    #     self.train_idx = train_idx[len(train_idx) // 5 : ]
    #     self.test_idx = data.test_idx
    #     # edges data process
    #     # edge_type = torch.from_numpy(data.edge_type).type(torch.LongTensor)
    #     # edge_norm = torch.from_numpy(data.edge_norm).type(torch.LongTensor).unsqueeze(1)
    #     edge_type = tf.convert_to_tensor(data.edge_type)
    #     edge_norm = tf.convert_to_tensor(data.edge_norm).unsqueeze(1)
    #     # edge_type = torch.from_numpy(data.edge_type)
    #     # edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)
    #     # create graph
    #     self.graph = DGLGraph().to(self.device)
    #     self.graph.add_nodes(data.num_nodes)
    #     self.graph.add_edges(data.edge_src, data.edge_dst)
    #     self.graph.edata.update({'rel_type': edge_type.to(self.device), 'norm': edge_norm.to(self.device)})
    #     # self.graph.edata.update({'rel_type': edge_type.to(self.device).type(torch.LongTensor), 'norm': edge_norm.to(self.device).type(torch.LongTensor)})


    # def train(self):

    #     pbar = tqdm(range(self.epoch))
    #     self.val_acc = 0
    #     self.Epoch += 1
    #     for _ in pbar:
    #         # Compute prediction error
    #         logits = self.model(self.graph)
    #         loss = self.loss_fn(logits.type(torch.FloatTensor)[self.train_idx], self.labels.type(torch.LongTensor)[self.train_idx])

    #         # Backpropagation
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    #         # train_acc = (logits.type(torch.FloatTensor)[self.train_idx].argmax(1) == self.labels[self.train_idx]).float().sum()
    #         train_acc = (logits[self.train_idx].argmax(1) == self.labels[self.train_idx]).float().sum()
    #         train_acc = train_acc / len(self.train_idx)
    #         # val_acc =  (logits.type(torch.FloatTensor)[self.val_idx].argmax(1) == self.labels[self.val_idx]).float().sum()
    #         val_acc =  (logits[self.val_idx].argmax(1) == self.labels[self.val_idx]).float().sum()
    #         val_acc = val_acc / len(self.val_idx)

    #         self.val_acc = self.val_acc + val_acc

    #         pbar.set_description("Client {:>2} Training: Train Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}".format(
    #                             self.client_id, loss.item(), train_acc, val_acc))
    #         self.writer.add_scalar(f"training/loss/{self.client_id}", loss.item(), self.Epoch * self.epoch + _)
    #         self.writer.add_scalar(f"training/acc/{self.client_id}", train_acc, self.Epoch * self.epoch + _)
    #         self.writer.add_scalar(f"val/acc/{self.client_id}", val_acc, self.Epoch * self.epoch + _)

    #     self.writer.add_embedding(logits[self.val_idx], self.labels[self.val_idx], global_step=self.Epoch, tag="clent"+str(self.client_id))

    #     self.model_param = self.model.state_dict()
    #     self.val_acc = self.val_acc / self.epoch

    # def test(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         logits = self.model(self.graph)
    #         test_loss = self.loss_fn(logits.type(torch.FloatTensor)[self.test_idx], self.labels.type(torch.LongTensor)[self.test_idx])
    #         test_acc =  (logits[self.test_idx].argmax(1) == self.labels[self.test_idx]).float().sum()
    #         test_acc = test_acc / len(self.test_idx)

    #     self.logging.info("Client {:>2} Test: Test Loss: {:.4f} | Test Acc: {:.4f}".format(
    #         self.client_id, test_loss.item(), test_acc
    #     ))
    #     # save to disk
    #     torch.save(self.model_param, self.model_path + "client" + str(self.client_id) + '_model.ckpt')

    #     return test_acc
    
    # # get the global model's parameters from parameter server
    # def getParame(self, round, param):
    #     self.round = round
    #     if self.model_param is not None:
    #         for layer_name in self.model_param:
    #             if ("layers.0" in layer_name) or layer_name.endswith("w_comp"): 
    #                 param[layer_name] = copy.deepcopy(self.model_param[layer_name])
    #     self.model.load_state_dict(param)


    # # upload the local model's parameters to parameter server
    # def uploadParame(self):
    #     param = {}
    #     for layer_name in self.model_param:
    #         if not layer_name.endswith("w_comp") and ( "layers.0" not in layer_name):
    #             param[layer_name] = self.model_param[layer_name]
    #     return self.round, param, self.val_acc