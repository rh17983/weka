# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# classifiers.py
# Copyright (C) 2014-2019 Fracpete (pythonwekawrapper at gmail dot com)

import os
import traceback
import weka.core.jvm as jvm
import helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier, \
    PredictionOutput, Kernel, KernelClassifier
from weka.classifiers import Evaluation
from weka.filters import Filter
from weka.core.classes import Random, from_commandline
import weka.plot.classifiers as plot_cls
import weka.plot.graph as plot_graph
import weka.core.typeconv as typeconv


def main():
    """
    Just runs some example code.
    """

    # load a dataset
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()

    # build a classifier and output model
    helper.print_title("Training classifier")

    classifier = Classifier(classname="weka.classifiers.trees.LMT")
    classifier.build_classifier(iris_data)

    print(classifier)
    print(classifier.graph)
    print(classifier.to_source("MyLMT"))
    plot_graph.plot_dot_graph(classifier.graph)

    # evaluate model on test set
    helper.print_title("Evaluating classifier on iris")

    evaluation = Evaluation(iris_data)
    evl = evaluation.test_model(classifier, iris_data)

    print(evl)
    print(evaluation.summary())

    # evaluate model on train/test split
    helper.print_title("Evaluating classifier on iris (random split 66%)")
    evaluation = Evaluation(iris_data)
    evaluation.evaluate_train_test_split(classifier, iris_data, 66.0, Random(1))
    print(evaluation.summary())


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
