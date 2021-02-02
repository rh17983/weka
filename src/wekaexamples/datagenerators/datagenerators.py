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

# datagenerators.py
# Copyright (C) 2014-2017 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from weka.datagenerators import DataGenerator


def main():
    """
    Just runs some example code.
    """

    helper.print_title("Generate data (Agrawal)")
    generator = DataGenerator(
        classname="weka.datagenerators.classifiers.classification.Agrawal",
        options=["-n", "10", "-r", "agrawal"])
    generator.dataset_format = generator.define_data_format()
    print(generator.dataset_format)
    if generator.single_mode_flag:
        for i in range(generator.num_examples_act):
            print(generator.generate_example())
    else:
        print(generator.generate_examples())

    helper.print_title("Generate data (BayesNet)")
    generator = DataGenerator(
        classname="weka.datagenerators.classifiers.classification.BayesNet",
        options=["-S", "2", "-n", "10", "-C", "10"])
    generator.dataset_format = generator.define_data_format()
    print(generator.dataset_format)
    if generator.single_mode_flag:
        for i in range(generator.num_examples_act):
            print(generator.generate_example())
    else:
        print(generator.generate_examples())

    # partial classname
    helper.print_title("Creating clusterer from partial classname")
    clsname = ".Agrawal"
    generator = DataGenerator(classname=clsname)
    print(clsname + " --> " + generator.classname)

if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
