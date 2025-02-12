# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import unittest
import os
import time
import json
import requests

class UnitTest(unittest.TestCase):
    def setUp(self) -> None:
        yaml_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests/nightly/server/askdoc.yaml"
        if os.path.exists(yaml_file_path):
            command = f'neuralchat_server start \
                        --config_file {yaml_file_path} \
                        --log_file "./neuralchat.log"'
        elif os.path.exists("./askdoc.yaml"):
            command = f'neuralchat_server start \
                                    --config_file ./askdoc.yaml \
                                    --log_file "./neuralchat.log"'
        else:
            command = 'sed -i "s|askdoc|ci/server/askdoc|g" ./nightly/server/askdoc.yaml && neuralchat_server start \
                        --config_file "./nightly/server/askdoc.yaml" \
                        --log_file "./neuralchat.log"'
        try:
            self.server_process = subprocess.Popen(command,
                                    universal_newlines=True, shell=True) # nosec
            time.sleep(20)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)

    def tearDown(self) -> None:
        # kill server process
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

        # delete created resources
        import shutil
        if os.path.exists("./out_persist"):
            shutil.rmtree("./out_persist")
        if os.path.exists("./photoai_retrieval_docs"):
            shutil.rmtree("./photoai_retrieval_docs")

    def test_askdoc_chat(self):
        # test chat API
        url = 'http://127.0.0.1:6001/v1/aiphotos/askdoc/chat'
        request = {
            "query": "oneAPI编译器是什么?",
            "translated": "What is Intel oneAPI Compiler?",
            "knowledge_base_id": "default",
            "stream": False,
            "max_new_tokens": 256
        }
        res = requests.post(url, json.dumps(request))
        self.assertEqual(res.status_code, 200)

        request = {
            "query": "蔡英文是谁?",
            "translated": "Who is Tsai Ing-wen?",
            "knowledge_base_id": "default",
            "stream": False,
            "max_new_tokens": 256
        }
        res = requests.post(url, json.dumps(request))
        self.assertEqual(res.status_code, 200)
        self.assertIn('Your query contains sensitive words, please try another query', str(res.text))

        # test create API
        url = 'http://127.0.0.1:6001/v1/aiphotos/askdoc/create'
        file_path = \
           "/intel-extension-for-transformers/intel_extension_for_transformers \
            /neural_chat/assets/docs/sample.txt"
        print("########", os.getcwd())

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                response = requests.post(url, files={"file": file})
        elif os.path.exists("../assets/docs/sample.txt"):
            with open("../assets/docs/sample.txt", 'rb') as file:
                response = requests.post(url, files={"file": file})
        else:
            with open("../../../assets/docs/sample.txt", 'rb') as file:
                response = requests.post(url, files={"file": file})
        print(response.text)
        self.assertEqual(response.status_code, 200)
        self.assertIn('knowledge_base_id', response.text.lower())
        kb_id = json.loads(response.text)['knowledge_base_id']

        # test append API
        url = 'http://127.0.0.1:6001/v1/aiphotos/askdoc/append'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                response = requests.post(
                    url, 
                    files={
                        "file": file,
                        "knowledge_base_id": (None, kb_id)})
        elif os.path.exists("../assets/docs/sample.txt"):
            with open("../assets/docs/sample.txt", 'rb') as file:
                response = requests.post(
                    url, 
                    files={
                        "file": file,
                        "knowledge_base_id": (None, kb_id)})
        else:
            with open("../../../assets/docs/sample.txt", 'rb') as file:
                response = requests.post(
                    url, 
                    files={
                        "file": file,
                        "knowledge_base_id": (None, kb_id)})
        print(response.text)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Succeed', response.text)

        # test upload_link API - create new kb
        url = 'http://127.0.0.1:6001/v1/aiphotos/askdoc/upload_link'
        data = {
            "link_list": ["https://www.ces.tech/"]
        }
        response = requests.post(url, json=data)
        print(response.text)
        self.assertEqual(response.status_code, 200)
        self.assertIn('knowledge_base_id', response.text)

        # test upload_link API - append
        data['knowledge_base_id'] = kb_id
        response = requests.post(url, json=data)
        print(response.text)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Succeed', response.text)


if __name__ == "__main__":
    unittest.main()
