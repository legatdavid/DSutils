# Databricks notebook source
class Metadata:
    _struct = pd.DataFrame(index=['name'],
                  columns=['type', 'level', 'role',
                           'min', 'max'],
                  dtype=['str', 'str', 'float', 'float']
                          )

    def __init__(self, data):
        print("TBD")

    def set_target(self, columns):
        print("TBD")

    def set_rejected(self, columns):
        print("TBD")

    def set_category(self, columns):
        print("TBD")

