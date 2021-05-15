# encoding: utf-8
"""Implementations of all on-path strategies"""
from __future__ import division
import random

import networkx as nx

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

__all__ = [
       'Partition',
       'Edge',
       'LeaveCopyEverywhere',
       'LeaveCopyDown',
       'ProbCache',
       'CacheLessForMore',
       'RandomBernoulli',
       'RandomChoice',
       'ContentPopularityNodeImportance',
       'ActivePushOffpath',
           ]


@register_strategy('PARTITION')
class Partition(Strategy):
    """Partition caching strategy.

    In this strategy the network is divided into as many partitions as the number
    of caching nodes and each receiver is statically mapped to one and only one
    caching node. When a request is issued it is forwarded to the cache mapped
    to the receiver. In case of a miss the request is routed to the source and
    then returned to cache, which will store it and forward it back to the
    receiver.

    This requires median cache placement, which optimizes the placement of
    caches for this strategy.

    This strategy is normally used with a small number of caching nodes. This
    is the the behaviour normally adopted by Network CDN (NCDN). Google Global
    Cache (GGC) operates this way.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Partition, self).__init__(view, controller)
        if 'cache_assignment' not in self.view.topology().graph:
            raise ValueError('The topology does not have cache assignment '
                             'information. Have you used the optimal median '
                             'cache assignment?')
        self.cache_assignment = self.view.topology().graph['cache_assignment']

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        source = self.view.content_source(content)
        self.controller.start_session(time, receiver, content, log)
        cache = self.cache_assignment[receiver]
        self.controller.forward_request_path(receiver, cache)
        if not self.controller.get_content(cache):
            self.controller.forward_request_path(cache, source)
            self.controller.get_content(source)
            self.controller.forward_content_path(source, cache)
            self.controller.put_content(cache)
        self.controller.forward_content_path(cache, receiver)
        self.controller.end_session()


@register_strategy('EDGE')
class Edge(Strategy):
    """Edge caching strategy.

    In this strategy only a cache at the edge is looked up before forwarding
    a content request to the original source.

    In practice, this is like an LCE but it only queries the first cache it
    finds in the path. It is assumed to be used with a topology where each
    PoP has a cache but it simulates a case where the cache is actually further
    down the access network and it is not looked up for transit traffic passing
    through the PoP but only for PoP-originated requests.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller):
        super(Edge, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        edge_cache = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                edge_cache = v
                if self.controller.get_content(v):
                    serving_node = v
                else:
                    # Cache miss, get content from source
                    self.controller.forward_request_path(v, source)
                    self.controller.get_content(source)
                    serving_node = source
                break
        else:
            # No caches on the path at all, get it from source
            self.controller.get_content(v)
            serving_node = v

        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        self.controller.forward_content_path(serving_node, receiver, path)
        if serving_node == source:
            self.controller.put_content(edge_cache)
        self.controller.end_session()


@register_strategy('LCE')
class LeaveCopyEverywhere(Strategy):
    """Leave Copy Everywhere (LCE) strategy.

    In this strategy a copy of a content is replicated at any cache on the
    path between serving node and receiver.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyEverywhere, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('LCD')
class LeaveCopyDown(Strategy):
    """Leave Copy Down (LCD) strategy.

    According to this strategy, one copy of a content is replicated only in
    the caching node you hop away from the serving node in the direction of
    the receiver. This strategy is described in [2]_.

    Rereferences
    ------------
    ..[1] N. Laoutaris, H. Che, i. Stavrakakis, The LCD interconnection of LRU
          caches and its analysis.
          Available: http://cs-people.bu.edu/nlaout/analysis_PEVA.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(LeaveCopyDown, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # Leave a copy of the content only in the cache one level down the hit
        # caching node
        copied = False
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if not copied and v != receiver and self.view.has_cache(v):
                self.controller.put_content(v)
                copied = True
        self.controller.end_session()


@register_strategy('PROB_CACHE')
class ProbCache(Strategy):
    """ProbCache strategy [3]_

    This strategy caches content objects probabilistically on a path with a
    probability depending on various factors, including distance from source
    and destination and caching space available on the path.

    This strategy was originally proposed in [2]_ and extended in [3]_. This
    class implements the extended version described in [3]_. In the extended
    version of ProbCache the :math`x/c` factor of the ProbCache equation is
    raised to the power of :math`c`.

    References
    ----------
    ..[2] I. Psaras, W. Chai, G. Pavlou, Probabilistic In-Network Caching for
          Information-Centric Networks, in Proc. of ACM SIGCOMM ICN '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/prob-cache-icn-sigcomm12.pdf
    ..[3] I. Psaras, W. Chai, G. Pavlou, In-Network Cache Management and
          Resource Allocation for Information-Centric Networks, IEEE
          Transactions on Parallel and Distributed Systems, 22 May 2014
          Available: http://doi.ieeecomputersociety.org/10.1109/TPDS.2013.304
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, t_tw=10):
        super(ProbCache, self).__init__(view, controller)
        self.t_tw = t_tw
        self.cache_size = view.cache_nodes(size=True)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        c = len([node for node in path if self.view.has_cache(node)])
        x = 0.0
        for hop in range(1, len(path)):
            u = path[hop - 1]
            v = path[hop]
            N = sum([self.cache_size[n] for n in path[hop - 1:]
                     if n in self.cache_size])
            if v in self.cache_size:
                x += 1
            self.controller.forward_content_hop(u, v)
            if v != receiver and v in self.cache_size:
                # The (x/c) factor raised to the power of "c" according to the
                # extended version of ProbCache published in IEEE TPDS
                prob_cache = float(N) / (self.t_tw * self.cache_size[v]) * (x / c) ** c
                if random.random() < prob_cache:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CL4M')
class CacheLessForMore(Strategy):
    """Cache less for more strategy [4]_.

    This strategy caches items only once in the delivery path, precisely in the
    node with the greatest betweenness centrality (i.e., that is traversed by
    the greatest number of shortest paths). If the argument *use_ego_betw* is
    set to *True* then the betweenness centrality of the ego-network is used
    instead.

    References
    ----------
    ..[4] W. Chai, D. He, I. Psaras, G. Pavlou, Cache Less for More in
          Information-centric Networks, in IFIP NETWORKING '12
          Available: http://www.ee.ucl.ac.uk/~uceeips/centrality-networking12.pdf
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, use_ego_betw=False, **kwargs):
        super(CacheLessForMore, self).__init__(view, controller)
        topology = view.topology()
        if use_ego_betw:
            self.betw = dict((v, nx.betweenness_centrality(nx.ego_graph(topology, v))[v])
                             for v in topology.nodes())
        else:
            self.betw = nx.betweenness_centrality(topology)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        # No cache hits, get content from source
        else:
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        # get the cache with maximum betweenness centrality
        # if there are more than one cache with max betw then pick the one
        # closer to the receiver
        max_betw = -1
        designated_cache = None
        for v in path[1:]:
            if self.view.has_cache(v):
                if self.betw[v] >= max_betw:
                    max_betw = self.betw[v]
                    designated_cache = v
        # Forward content
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_BERNOULLI')
class RandomBernoulli(Strategy):
    """Bernoulli random cache insertion.

    In this strategy, a content is randomly inserted in a cache on the path
    from serving node to receiver with probability *p*.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, p=0.2, **kwargs):
        super(RandomBernoulli, self).__init__(view, controller)
        self.p = p

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v != receiver and self.view.has_cache(v):
                if random.random() < self.p:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('RAND_CHOICE')
class RandomChoice(Strategy):
    """Random choice strategy

    This strategy stores the served content exactly in one single cache on the
    path from serving node to receiver selected randomly.
    """

    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(RandomChoice, self).__init__(view, controller)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
        else:
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        caches = [v for v in path[1:-1] if self.view.has_cache(v)]
        designated_cache = random.choice(caches) if len(caches) > 0 else None
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if v == designated_cache:
                self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('CPNI')
class ContentPopularityNodeImportance(Strategy):
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller, **kwargs):
        super(ContentPopularityNodeImportance, self).__init__(view, controller)
        topology = view.topology()
        self.request = {}                #拓扑内节点的请求次数
        self.req_list = {}
        for i in view.topology().nodes():
            self.req_list[i] = {}  #每一节点的内容流行度表初始化为空
            self.request[i] = 0         #所有节点请求次数初始化为0
        
        self.betw = nx.betweenness_centrality(topology)    #所有节点的中介中心性
        max_betw = max(self.betw.values())
        for i in self.betw:
            self.betw[i] = self.betw[i] / max_betw
        
        self.deg = nx.degree_centrality(topology)          #所有节点的介数
        max_deg = max(self.deg.values())
        for i in self.deg:
            self.deg[i] = self.deg[i] / max_deg
        
        self.close = nx.closeness_centrality(topology)          #所有节点的紧密中心性
        max_close = max(self.close.values())
        for i in self.close:
            self.close[i] = self.close[i] / max_close
        
        self.nodeweight = {}
        for i in self.betw:
            self.nodeweight[i] = (self.betw[i] ** 2 + self.deg[i] ** 2 + self.close[i] ** 2) ** 0.5
            self.nodeweight[i] = self.nodeweight[i] / 3

        # max_node = max(self.nodeweight.values())
        # for i in self.nodeweight:
        #     self.nodeweight[i] = self.nodeweight[i] / max_node
        
        self.distance = dict(nx.all_pairs_dijkstra_path_length(topology, weight='delay'))    #拓扑内的任俩节点间距离列表

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)           #得到content源节点的位置  
        locations = self.view.content_locations(content)     #得到所有缓存content副本的节点（含源节点）
        
        nearest_replica = min(locations, key=lambda x: self.distance[receiver][x])
        path = self.view.shortest_path(receiver, nearest_replica) 
        
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        cacheNode = []    
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v):
                self.request[v] += 1
                if content in self.req_list[v]:
                    self.req_list[v][content] += 1
                else:
                    self.req_list[v][content] = 1    


                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.req_list[v][content] > self.req_list[v][cache] :
                                cacheNode.append(v)
                                break
                    else:
                        cacheNode.append(v)
        else:    
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)
        self.controller.end_session()


@register_strategy('APOP')
class ActivePushOffpath(Strategy):
    
    @inheritdoc(Strategy)
    def __init__(self, view, controller, radius = 3, **kwargs):
        super(ActivePushOffpath, self).__init__(view, controller)
        topology = view.topology()
        self.total_req = 0
        self.request = {}                #拓扑内节点的请求次数
        self.req_list = {}
        self.content_req = {}
        self.content_req_ratio = {}
        for i in view.topology().nodes():
            self.req_list[i] = {}  #每一节点的内容流行度表初始化为空
            self.request[i] = 0         #所有节点请求次数初始化为0
        
        self.betw = nx.betweenness_centrality(topology)    #所有节点的中介中心性
        max_betw = max(self.betw.values())
        for i in self.betw:
            self.betw[i] = self.betw[i] / max_betw
        
        self.deg = nx.degree_centrality(topology)          #所有节点的介数
        max_deg = max(self.deg.values())
        for i in self.deg:
            self.deg[i] = self.deg[i] / max_deg
        
        self.close = nx.closeness_centrality(topology)          #所有节点的紧密中心性
        max_close = max(self.close.values())
        for i in self.close:
            self.close[i] = self.close[i] / max_close
        
        self.nodeweight = {}
        for i in self.betw:
            self.nodeweight[i] = (self.betw[i] ** 2 + self.deg[i] ** 2 + self.close[i] ** 2) ** 0.5
            self.nodeweight[i] = self.nodeweight[i] / 3

        max_node = max(self.nodeweight.values())
        for i in self.nodeweight:
            self.nodeweight[i] = self.nodeweight[i] / max_node
        
        self.distance = dict(nx.all_pairs_dijkstra_path_length(topology, weight='delay'))    #拓扑内的任俩节点间距离列表

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get all required data
        source = self.view.content_source(content)           #得到content源节点的位置  
        locations = self.view.content_locations(content)     #得到所有缓存content副本的节点（含源节点）
        
        nearest_replica = min(locations, key=lambda x: self.distance[receiver][x])
        path = self.view.shortest_path(receiver, nearest_replica) 
        
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log)
        cacheNode = []    
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)            
            if self.view.has_cache(v):
                self.total_req += 1
                self.request[v] += 1
                if content in self.req_list[v]:
                    self.req_list[v][content] += 1
                else:
                    self.req_list[v][content] = 1

                if content in self.content_req:
                    self.content_req[content] += 1
                else:
                    self.content_req[content] = 1


                if self.controller.get_content(v):
                    serving_node = v
                    break
                else:                   
                    if self.view.is_cache_full(v):
                        for cache in self.view.cache_dump(v):
                            if self.req_list[v][content] > self.req_list[v][cache] :
                                cacheNode.append(v)
                                break
                    else:
                        cacheNode.append(v)
        else:    
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                if v in cacheNode:
                    self.controller.put_content(v)                    
                    temp = sorted(self.content_req.items(), key=lambda item:item[1], reverse = True)
                    if(self.content_req[content] > temp[int(len(temp) / 4)][1]):
                        neighbor = self.controller.get_neighbors(v)
                        weight = 0
                        candidate = v
                        for node in neighbor:
                            if self.nodeweight[node] > weight:
                                weight = self.nodeweight[node]
                                candidate = node
                        if self.nodeweight[candidate] > self.nodeweight[v]:
                            if (self.view.cache_lookup(candidate, content) == False):
                                self.controller.forward_content_hop(v, candidate)
                                if self.view.is_cache_full(candidate):
                                    for cache in self.view.cache_dump(candidate):
                                        if self.content_req[content] > self.content_req[cache]:
                                            self.total_req += 1
                                            self.request[candidate] += 1
                                            if content in self.req_list[candidate]:
                                                self.req_list[candidate][content] += 1
                                            else:
                                                self.req_list[candidate][content] = 1                              
                                            self.controller.put_content(candidate)
                                            break
                                else:
                                    self.total_req += 1
                                    self.request[candidate] += 1
                                    if content in self.req_list[candidate]:
                                        self.req_list[candidate][content] = self.req_list[candidate][content] + 1
                                    else:
                                        self.req_list[candidate][content] = 1
                                    self.controller.put_content(candidate)
        self.controller.end_session()