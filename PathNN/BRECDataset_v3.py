import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx, to_networkx
import os
from tqdm import tqdm
import igraph as ig

torch_geometric.seed_everything(2022)

def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))

class ModifData(Data):
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
        super().__init__(x=x, edge_index=edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key or "path" in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0

class BRECDataset(InMemoryDataset):
    def __init__(self, name="no_param", root="Data", transform=None, pre_transform=None, pre_filter=None, cutoff=2, path_type='shortest_path', undirected=True):
        self.root = root
        self.name = name
        self.cutoff = cutoff
        self.path_type = path_type
        self.undirected = undirected
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]

    def process(self):
        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        processed_data_list = [self._add_paths(data) for data in tqdm(data_list)]
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _add_paths(self, data):
        G = ig.Graph.from_networkx(to_networkx(data, to_undirected=self.undirected))
        graph_info = fast_generate_paths2(G, self.cutoff, self.path_type, undirected=self.undirected)
        data = ModifData(**data.stores[0])
        data.edge_index = data.edge_index

        for jj in range(1, self.cutoff - 1):
            paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
            if paths.size(0) > 0:
                setattr(data, f'path_{jj + 2}', paths)
            else:
                setattr(data, f'path_{jj + 2}', torch.empty(0, jj + 2).long())
        return data

def fast_generate_paths2(g, cutoff, path_type, weights=None, undirected=True):
    if undirected and g.is_directed():
        g.to_undirected()

    path_length = np.array(g.distances())
    if path_type != "all_simple_paths":
        diameter = g.diameter(directed=False)
        diameter = diameter + 1 if diameter + 1 < cutoff else cutoff
    else:
        diameter = cutoff

    X = [[] for i in range(cutoff - 1)]
    sp_dists = [[] for i in range(cutoff - 1)]

    for n1 in range(g.vcount()):
        if path_type == "all_simple_paths":
            paths_ = g.get_all_simple_paths(n1, cutoff=cutoff - 1)
            for path in paths_:
                idx = len(path) - 2
                if len(path) > 0:
                    X[idx].append(path)
                    sp_dist = []
                    for node in path:
                        sp_dist.append(path_length[n1, node])
                    sp_dists[idx].append(sp_dist)
        else:
            valid_ngb = [i for i in np.where((path_length[n1] <= cutoff - 1) & (path_length[n1] > 0))[0] if i > n1]
            for n2 in valid_ngb:
                if path_type == "shortest_path":
                    paths_ = g.get_shortest_paths(n1, n2, weights=weights)
                elif path_type == "all_shortest_paths":
                    paths_ = g.get_all_shortest_paths(n1, n2, weights=weights)
                for path in paths_:
                    idx = len(path) - 2
                    X[idx].append(path)
                    X[idx].append(list(reversed(path)))

    return X, diameter, sp_dists

def main():
    dataset = BRECDataset(cutoff=3, path_type='shortest_path')
    print(len(dataset))

if __name__ == "__main__":
    main()
