# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:58:03 2024

@author: Ryan.Larson
"""

from pyupdater.client.plugins import GitHubUpdater

class ClientConfig(object):
    PUBLIC_KEY = "YourPublicKey"  # You'll get this after signing your updates later

    def init(self):
        self.data_dir = "appdata"  # Local storage for update files
        self.company_name = "Ryan Larson"
        self.update_urls = ["https://github.com/rockwell-window-wells/mark10-processing/releases"]  # URL of your GitHub releases
        self.max_download_retries = 3
        self.plugin = GitHubUpdater  # Using GitHub for hosting
