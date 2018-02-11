
#####Setting up cuda and cudnn in user directory######

#Copy cuda sdk from root to user directory
cp -r /usr/local/cuda-8.0 {HOME}/program/

#Download cudNN 5.0 from nvidia website
#I was not able to directly copy using wget, if you guys can then do it
#I copied it in my local machine and did scp
#Steps are below
#The link for cudnn might be different, just test it
#Uncomment below lines

#sudo scp -i "id_rsa" -o "RSAAuthentication=yes" /home/atit/cudnn-8.0* akshetty@arc10.csc.ncsu.edu:/home/akshetty

#Or you can use
#curl -O http://developer.download.nvidia.com/compute/redist/cudnn/v2/cudnn-8.0-linux-x64-v5.1.tgz

#Once copied, extract it using:
#tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
#Alternatively you can do a general unzip
#gunzip cudnn-8.0-linux-x64-v5.1.tgz

#Then do following

cp -P cuda/include/cudnn.h /home/akshetty/program/cuda-8.0/include/
cp -P cuda/lib64/libcudnn* /home/akshetty/program/cuda-8.0/lib64/

#The above steps will make create local copy of which we can modify, else we would've required admin access

#####Run and Put following in .bash_profile, so that you won't have to put this again and again#####

export PATH=$HOME/bin:$PATH 
export LD_LIBRARY_PATH=$HOME/lib/:$LD_LIBRARY_PATH

export PATH=${HOME}/program/cuda-8.0:$PATH
export LD_LIBRARY_PATH=${HOME}/program/cuda-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=${HOME}/program/cuda-8.0c/cuda/include:$CPATH
export LIBRARY_PATH=${HOME}/program/cuda-8.0/lib64:$LIBRARY_PATH

#Open ~/.bash_profile using any editor eg
vi ~/.bash_profile
#Then edit and save. 
#################################


#####Install Pip#####
if [[ ! -x ~/.local/bin/pip ]]; then
  # get pip
  curl https://bootstrap.pypa.io/get-pip.py > get-pip.py
  python -W ignore get-pip.py --user
fi
# pip is installed into ~/.local/bin so add that to PATH so it's available
PATH=$PATH:~/.local/bin
 
# now install/upgrade required python modules
pip install --upgrade --user bugzilla bugzillatools python-bugzilla jira pytz

############################################################

#####Install Other Pip packages#####
pip install --user Cython --install-option="--no-cython-compile"

pip install --user nose

pip install --user Pillow

pip install --user setuptools --upgrade

pip install --user bcolz

pip install --user --upgrade sklearn

#####Install cmake####
wget https://cmake.org/files/v3.8/cmake-3.8.0-rc4.tar.gz
tar xvzf cmake*.tar.gz
cd cmake*
./configure --prefix=$HOME

make

make install

######################

#####Install Theano#####
pip install --user Theano

#############################

#libgpuarray helps Theano use gpu
#####Install libpuarray#####
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
git checkout tags/v0.6.2 -b v0.6.2

rm -rf build Build
mkdir Build
cd Build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release
make
make install


cd ..

# Run the following export and add them in your ~/.bash_profile file
export CPATH=$CPATH:~/.local/include
export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib
#Go inside libgpuarray folder
python setup.py build
python setup.py install --user
cd
DEVICE="cuda" python -c "import pygpu;pygpu.test()"

################################################


#####Install Keras#####
pip install --user keras


#Keras will by default use Tensorflow as backend
#Change that configuration
vi $HOME/.keras/keras.json

#Edit the value of "backend" to "theano"
#######################


####Environment variable required for Theano#####
#You should set some config for Theano so that it will use the gpu, otherwise default is cpu
#Run and Put below in .bash_profile

export THEANO_FLAGS='device=cuda,floatX=float32,dnn.enabled=True,force_device=True'

#dnn.enabled to enable cudnn
#force_deviceto enable gpu, else throw error
#device tells which gpu to use, if more than one gpu is there, eg cuda0, cuda1
#many more configs can be set here

export CUDA_ROOT=$HOME/program/cuda-8.0

#Tells Theano to use local cuda, else it will use default cuda located at usr/lib/cuda

###########################################



#Run below python code to check whether the Theano is using gpu

from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
