# Databricks notebook source
from DSutils.deploy.scobot import ScoBot

class TestScoBot:
    
    def test_get_name(self):
        sb = ScoBot("Karel", "0.1.0")
        assert sb.get_name() == "Karel"
    
    
    def test_get_version(self):
        sb = ScoBot("Karel", "0.1.0")
        assert sb.get_version() == "0.1.0"
