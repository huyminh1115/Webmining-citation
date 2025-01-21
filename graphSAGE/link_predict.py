import itertools

import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs

# MODEL DEFINITIONS
from dgl.nn import SAGEConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")
        self.conv3 = SAGEConv(h_feats, h_feats, "mean")
        self.conv4 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        return h


import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


if __name__ == "__main__":
    # graph_data = "./citation_541_384.dgl"
    # graph_data = "./citation_541_768.dgl"
    # graph_data = "./citation_6897_384.dgl"
    graph_data = "./citation_6897_768.dgl"
    glist, label_dict = load_graphs(graph_data)  # glist will be [g1, g2]
    g = glist[0]
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # Create different graphs for training and testing
    train_g = dgl.remove_edges(g, eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    h_feats = 128
    writer = SummaryWriter(log_dir=f"logs/{graph_data}_{h_feats}")
    model = GraphSAGE(train_g.ndata["feat"].shape[1], h_feats)
    # pred = DotPredictor()
    pred = MLPPredictor(h_feats)

    def compute_loss(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        precision = precision_score(
            torch.cat(
                [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
            ).numpy(),
            torch.cat([pos_score, neg_score]).numpy() > 0,
        )
        recall = recall_score(
            torch.cat(
                [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
            ).numpy(),
            torch.cat([pos_score, neg_score]).numpy() > 0,
        )
        accuracy = accuracy_score(
            torch.cat(
                [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
            ).numpy(),
            torch.cat([pos_score, neg_score]).numpy() > 0,
        )
        return roc_auc_score(labels, scores), precision, recall, accuracy

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.001
    )
    n_epochs = 1000
    for e in range(n_epochs):
        h = model(train_g, train_g.ndata["feat"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            print("In epoch {}, loss: {}".format(e, loss))
            with torch.no_grad():
                pos_score = pred(test_pos_g, h)
                neg_score = pred(test_neg_g, h)
                roc, precision, recall, accuracy = compute_auc(pos_score, neg_score)
                writer.add_scalar("Loss/train", loss, e)
                writer.add_scalar("ROC/train", roc, e)
                writer.add_scalar("Precision/train", precision, e)
                writer.add_scalar("Recall/train", recall, e)
                writer.add_scalar("Accuracy/train", accuracy, e)
