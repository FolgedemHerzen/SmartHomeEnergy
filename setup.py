from setuptools import setup, find_packages

setup(name='SmartHomeScheduling', version='1.0', author='FolgedemHerzen',
      author_email='zh7n21@soton.ac.uk', packages=find_packages(),
      install_requires=['numpy~=1.21.2', 'scipy~=1.8.0', 'pandas~=1.4.1'
                        , 'scikit-learn~=1.0.2',
                        'tensorflow~=2.8.0', 'keras~=2.8.0'],
      description='Smart Home Energy scheduling and '
                  'detection of abnormal pricing', long_description=open(
        'README.md').read())
