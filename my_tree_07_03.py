import os
import math
import datetime
import numpy as np
import pandas as pd
import pylab as plt
import random as rd
import pydotplus as pydot
from operator import itemgetter

#########################################################################################################
#########################################################################################################
#########################################################################################################

class Node():
    
    def __init__(self,tree,name,parent,index,method,custom_split_var_index=None,custom_split_var_value = None):
        self.name = name
        self.parent = parent
        self.index = index
        tree.nodes.append(self)
        tree.names.append(self.name)
        print("Initiated : {0}".format(self.name))
        if method == 'normal':
            self.__return_split_nodes(tree)
        else:
            self.__return_split_nodes_custom(tree,custom_split_var_index,custom_split_var_value)

    def __return_split_nodes(self,tree):
        temp_dat = [mod_dat[i] for i in self.index]
        current_target = [x[-1] for x in temp_dat]
        self.zeros = current_target.count(0)
        self.ones = current_target.count(1)
        self.population = self.ones + self.zeros
        self.depth = self.__find_depth(tree)
        
        if ((self.population > min_size) and (self.depth < max_depth)):
            best_split = self.__find_best_split(temp_dat)
            if len(best_split) > 0:
                left_index = [int(x[0]) for x in best_split[0]]
                right_index = [int(x[0]) for x in best_split[1]]
                self.split_var_index = best_split[2]
                self.split_var_value = best_split[3]
                self.gini = best_split[4]
                temp_node_left = Node(tree,max(tree.names) + 1,self.name,left_index,'normal')
                temp_node_right = Node(tree,max(tree.names) + 1,self.name,right_index,'normal')
                self.children = [temp_node_left,temp_node_right]
            else:
                self.split_var_index = None
                self.split_var_value = None
                self.gini = -.999
                self.children = []
        else:
            self.split_var_index = None
            self.split_var_value = None
            self.gini = -.999
            self.children = []

    def __return_split_nodes_custom(self,tree,custom_split_var_index,custom_split_var_value):
        temp_dat = [mod_dat[i] for i in self.index]
        current_target = [x[-1] for x in temp_dat]
        self.zeros = current_target.count(0)
        self.ones = current_target.count(1)
        self.population = self.ones + self.zeros
        self.depth = self.__find_depth(tree)
        
        if ((self.population > min_size) and (self.depth < max_depth)):
            best_split = self.__find_custom_split(temp_dat,custom_split_var_index,custom_split_var_value)
            if len(best_split) > 0:
                left_index = [int(x[0]) for x in best_split[0]]
                right_index = [int(x[0]) for x in best_split[1]]
                self.split_var_index = best_split[2]
                self.split_var_value = best_split[3]
                self.gini = best_split[4]
                temp_node_left = Node(tree,max(tree.names) + 1,self.name,left_index,'normal')
                temp_node_right = Node(tree,max(tree.names) + 1,self.name,right_index,'normal')
                self.children = [temp_node_left,temp_node_right]
            else:
                self.split_var_index = None
                self.split_var_value = None
                self.gini = -.999
                self.children = []
        else:
            self.split_var_index = None
            self.split_var_value = None
            self.gini = -.999
            self.children = []

    def __find_depth(self,tree):
        tt = 0
        for node in tree.nodes:
            if self.parent != None:
                if self.parent == node.name:
                    tt += 1
                    tt += node.depth
            else:
                break
        return tt
    
    def __find_best_split(self,temp_dat):
        max_gini = 0
        best_split = []
        for i in range(nx):
            split_dat_index = [0,i+1,-1]
            split_x_dat = [[x[i] for i in split_dat_index] for x in temp_dat]
            # split_x_dat = sorted(split_x_dat, key=itemgetter(1))
            # alternate way to sort the split_x_dat
            # split_x_dat.sort(key=lambda x: x[1])
            x_split_val_all = self.__make_x_split_val(split_x_dat)
            for x_split_val in x_split_val_all:
                left,right = self.__make_left_right(x_split_val,split_x_dat)
                temp_gini = self.__gini_index(left,right,[0,1],split_x_dat)
                if temp_gini > max_gini:
                    max_gini = temp_gini
                    best_split = [left,right,col_names[i],x_split_val,temp_gini]
        return best_split

    def __find_custom_split(self,temp_dat,custom_split_var_index,custom_split_var_value):
        max_gini = 0
        best_split = []
        i = col_names.get_loc(custom_split_var_index)
        split_dat_index = [0,i+1,-1]
        split_x_dat = [[x[i] for i in split_dat_index] for x in temp_dat]

        left,right = self.__make_left_right(custom_split_var_value,split_x_dat)
        temp_gini = self.__gini_index(left,right,[0,1],split_x_dat)
        if temp_gini > max_gini:
            max_gini = temp_gini
            best_split = [left,right,custom_split_var_index,custom_split_var_value,temp_gini]
        return best_split

    def __make_left_right(self,x_split_val,split_x_dat):
        left = []
        right = []
        for i in split_x_dat:
            if i[1] <= x_split_val:
                left.append(i)
            else:
                right.append(i)
        return left,right

    def __make_x_split_val(self,split_x_dat):
        x_temp = [x[1] for x in split_x_dat]
        x_temp = np.unique(x_temp).tolist()
        if len(x_temp) > k:
            return [np.percentile(x_temp,i) for i in np.linspace(0,100,k).tolist()[1:-1]]
        else:
            return x_temp
        

    def __gini_index(self,left,right,class_values,split_dat):
        gini2 = 0.0
        gini3 = 0.0
        gini_node_cmpl = 0.0
        sizeL = len(left)
        sizeR = len(right)
         
        for class_value in class_values:
            gini_node_cmpl += math.pow([row[-1] for row in split_dat].count(class_value) / float(len(split_dat)),2)                        
            if sizeL == 0:
                proportionL = 0
            else:
                proportionL = [row[-1] for row in left].count(class_value) / float(sizeL)

            if sizeR == 0:
                proportionR = 0
            else:
                proportionR = [row[-1] for row in right].count(class_value) / float(sizeR)
                    
            gini2 += math.pow(proportionL,2) * (sizeL/(sizeL + sizeR)) + math.pow(proportionR,2) * (sizeR/(sizeL + sizeR)) 

        gini3 = gini2 - gini_node_cmpl
        return gini3


class Tree():
    global mod_dat
    global col_names
    def __init__(self):
        self.nodes = []
        self.names = []
        
    def build_tree(self):
        temp_node = Node(self,0,None,[int(x[0]) for x in mod_dat],'normal')

    def remove_node(self,node_name):
        for node in self.nodes:
            if node.name == node_name:
                self.nodes.remove(node)
                self.names.remove(node_name)
                for child in node.children:
                     self.remove_node(child.name)

    def edit_node(self,node_name,custom_split_index,custom_split_value):
        for node in self.nodes:
            if node.name == node_name:
                temp_node = node
                break
        self.remove_node(node_name)
        if self.names:
            Node(self,max(self.names)+1,temp_node.parent,temp_node.index,'custom',custom_split_index,custom_split_value)
        else:
            Node(self,0,temp_node.parent,temp_node.index,'custom',custom_split_index,custom_split_value)

    def plot_tree(self,plot_name):
        graph = pydot.Dot(graph_type='graph')
        for parent_node in self.nodes:
            # if i see each node as parent, i need to find its children
            for child_node in self.nodes:
                if parent_node.name == child_node.parent:
                    # less than is always on the left
                    p = pydot.Node(name = parent_node.name,label = self.__make_text(parent_node), style="filled", fillcolor=self.__get_color(parent_node))
                    c = pydot.Node(name = child_node.name,label = self.__make_text(child_node), style="filled",fillcolor=self.__get_color(child_node))
                    graph.add_node(p)
                    graph.add_node(c)
                    graph.add_edge(pydot.Edge(p,c))
        graph.write_pdf(plot_name + '.pdf')

    def print_tree(self):
        print("\n--------------------------------------------------------------------------------")
        print("Printing Tree information now.")
        print("--------------------------------------------------------------------------------")
        for node in self.nodes:
            print('Name : {0}, Parent : {1}, Split Variable : {2}, Split Value = {3}, Gini = {4:.3f}, Population : {6}, Ones : {7}, Zeros : {8}'
                  .format(node.name,node.parent,node.split_var_index,node.split_var_value,node.gini,node.index,node.population,node.ones,node.zeros))

    def __make_text(self,node):
        if node.gini > 0:
            tt = 'Name : {0}, Parent : {1}, \nSplit at {2}<={3}, Gini = {4:.3f}, \nPopulation : {6}, Ones : {7}, Zeros : {8}'.format(node.name,node.parent,node.split_var_index,node.split_var_value,node.gini,node.index,node.population,node.ones,node.zeros)
            if node.parent == None:
                tt = 'Name : {0},\nSplit at {2}<={3}, Gini = {4:.3f}, \nPopulation : {6}, Ones : {7}, Zeros : {8}'.format(node.name,node.parent,node.split_var_index,node.split_var_value,node.gini,node.index,node.population,node.ones,node.zeros)
        else:
            tt = 'Name : {0}, Parent : {1}, \nPopulation : {6}, Ones : {7}, Zeros : {8}'.format(node.name,node.parent,node.split_var_index,node.split_var_value,node.gini,node.index,node.population,node.ones,node.zeros)
        return tt
    def __get_color(self,node):
        return 'gray' if node.zeros > node.ones else 'cyan'
        

    
#########################################################################################################
#########################################################################################################
#########################################################################################################

# 0 is index, -1 is y. Xs are in between. nx to be specified
global k
global nx
global mod_dat
global max_depth
global col_names
global min_size
global max_depth
global min_leaf_size

k=100

##mod_excel = pd.read_csv(os.getcwd()+"/data/FeatureDat_DT_V4.csv")
mod_excel = pd.read_excel(os.getcwd()+'/data/DT_data.xlsx',sheet_name = 'dummy_data')
mod_excel = mod_excel.dropna().reset_index().drop('index',1)
nx = mod_excel.shape[1]-1
mod_dat = mod_excel.values.tolist()
q = 0
for row in mod_dat:
        row.insert(0,q)
        q = q + 1

col_names = mod_excel.columns[0:-1]
print(mod_excel.shape)
print(col_names)
min_size = int(mod_excel.shape[0]/10)
min_leaf_size = int(mod_excel.shape[0]/50)
max_depth = 3

start_time = datetime.datetime.now()
print(("Start Date & Time: %s-%s-%s %s:%s:%s")%(start_time.day, start_time.month, start_time.year, start_time.hour, start_time.minute, start_time.second)) 
tree = Tree()
tree.build_tree()
end_time = datetime.datetime.now()
print(("End Date & Time: %s-%s-%s %s:%s:%s")%(end_time.day, end_time.month, end_time.year, end_time.hour, end_time.minute, end_time.second)) 
print(("Time Taken %s")%(end_time - start_time))

tree.plot_tree('my_tree_original')
##tree.edit_node(0,'height',130)
##tree.plot_tree('my_tree_edited')


#########################################################################################################
#########################################################################################################
#########################################################################################################
