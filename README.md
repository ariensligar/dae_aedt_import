

# Setting Up Python Environment

Here is one way to get everything up and running. 
Other options exists, one area that you may want to consider changing is the python envionrment manager. It doesn't matter what evionrment manager you use, as long as you can install all the required packages

### Python environment manager
Install anaconda:
https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe

# Creating Environment

'''
conda create -n dae_import python=3.8
activate dae_import 
'''

# Installing Packages
Install packages, by typing the following in your anaconda prompt
browse to directory where your dae scripts exists= and run

'''
pip install -r requirements.txt
'''

# Configuring IDE
You can use any IDE, if you want to use Spyder, 
install the windows standalone applications from: https://docs.spyder-ide.org/current/installation.html
you will also need to install, 

'''
pip install spyder-kernels
'''
note, depending on the version of spyder, you may need a specific version of spyder-kernals. 
For example Spyder 5.2 needs
'''
pip install spyder-kernels==2.2.0
'''

once this is installed, open up spyder and go to the menu Tool> Preferences > Python Interpreter
set the python interpreter to the one used in your anaconda environment, something like...

'''
C:\Users\<username>\Anaconda3\envs\5g_wizard\python.exe
'''

within spyder you can run Main.py

# DAE Files
you can access a library of DAE files from mixamo.com