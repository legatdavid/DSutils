# Databricks notebook source
class ScoBot:
    _model_name = "model"
    _model_version = "1.0.0"
    _steps = {}

    def __init__(self, name, version):
        self._model_name = name
        self._model_version = version

    def get_name(self):
        return self._model_name

    def get_version(self):
        return self._model_version

    def store(self, file=None):
        print("stored")
        print("Test GIT")
