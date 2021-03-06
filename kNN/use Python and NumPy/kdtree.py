"""
kd tree implementation:
essentially, the kd tree is a data structure to represent all the data,
"""
import math, operator
import numpy as np
from os import listdir
from kNN import img2vector

class KDNode:
    """

    """
    def __init__(self, point=None, split=None, LL=None, RR=None, label=None):
        """

        :param point: data point
        :param split: split
        :param LL: left child
        :param RR:  right child
        """

        self.point = point
        self.split = split
        self.left = LL
        self.right = RR
        self.label = label


def computeVariance(data_list):
    """
    compute the variance in the data_list
    :param data_list:
    :return:
    """
    for data in data_list:
        data = float(data)

    data_len = len(data_list)
    array = np.array(data_list)
    sum = array.sum()
    mean = sum / data_len
    array2 = array * array
    sum2 = array2.sum()
    mean2 = sum2 / data_len
    return mean2 -mean**2


def computeDist(data1, data2):
    """
    compute the distance, use E-distance
    :param data1:
    :param data2:
    :return:
    """
    sum = 0.0
    for i in range(len(data1)):
        sum += (data1[i] - data2[i]) * (data1[i] - data2[i])
    return math.sqrt(sum)


def createKDTree(root, data_list, label_list):
    """
    create the KD Tree
    :param root:
    :param data_list:
    :return:
    """
    data_len = len(data_list)
    if data_len == 0:
        return

    # choose the dimension with large variance
    dimension = len(data_list[0])
    #print "data size: " + str(data_len) + " dimension: " + str(dimension)

    split_dim = 0
    max_var = 0
    for i in range(dimension):
        ll = []
        for t in data_list:
            ll.append(t[i])

        var = computeVariance(ll)
        if var > max_var:
            max_var = var
            split_dim = i

    data_index = data_list[:, split_dim].argsort()
    label_list = np.array(label_list)
    data_list= data_list[data_index]

    label_list = label_list[data_index]

    point = data_list[data_len/2]
    label = label_list[data_len/2];

    root = KDNode(point, split=split_dim, label=label)
    root.left = createKDTree(root.left, data_list[0:data_len/2], label_list[0: data_len/2])
    root.right = createKDTree(root.right, data_list[data_len/2+1:data_len], label_list[data_len/2+1:data_len])
    return root


def findNN(root, query, K=1):
    """
    find the nearest point & dist
    :param root:
    :param query:
    :return: the nearest Node as well as the distance
    """

    min_dist = np.array([computeDist(query, root.point)])
    sorted_indices = min_dist.argsort()
    temp_root = root
    curr_point = [root]
    node_list = []

    while temp_root:
        node_list.append(temp_root)
        dist = computeDist(query, temp_root.point)

        if len(curr_point) == K:
            large_index = sorted_indices[-1]
            if min_dist[large_index] > dist:
                min_dist[large_index] = dist
                curr_point[large_index] = temp_root
                sorted_indices = min_dist.argsort()

        ss = temp_root.split
        if query[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right

    while node_list:
        node = node_list.pop()
        ss = node.split

        min_index = sorted_indices[-1]

        if abs(query[ss] - node.point[ss]) < min_dist[min_index]:
            if query[ss] <= node.point[ss]:
                temp_root = node.right
            else:
                temp_root = node.left

            if temp_root:
                node_list.append(temp_root)
                curr_dist = computeDist(query, temp_root.point)

                if min_dist[min_index] > curr_dist:
                    min_dist[min_index] = curr_dist
                    curr_point[min_index] = temp_root
                    sorted_indices = min_dist.argsort()

    return curr_point, min_dist


def KNN(root, K, target):
    """

    :param root:
    :param K:
    :param label:
    :return:
    """
    # get maximum count of curr_point labels
    curr_point, _ = findNN(root, target, K)
    count = {}
    for point in curr_point:
        if count.has_key(point.label):
            count[point.label] += 1
        else:
            count[point.label] = 1

    sortedCount = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]

def print_kdtree(root):
    """
    print the kd tree
    :param root:
    :return:
    """
    if root.left:
        print " ",
        print_kdtree(root.left)
    if root.right:
        print " ",
        print_kdtree(root.right)


def mainTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # build the kdtree
    print "start to build kd tree ..."
    print hwLabels
    kd_tree = KDNode()
    kd_tree = createKDTree(kd_tree, trainingMat, hwLabels)
    print "kd tree finished"
    #print_kdtree(kd_tree)
    #exit(0)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = KNN(kd_tree, 5, vectorUnderTest[0])

        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))


# note: kd-tree for the handwriting is not good enough, the dimension is higher than the number of the samples.
mainTest()




