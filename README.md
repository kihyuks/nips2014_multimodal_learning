#Improved Multimodal Deep Learning with Variation of Information<br />
######Kihyuk Sohn, Wenling Shang and Honglak Lee, NIPS 2014 [[pdf](http://papers.nips.cc/paper/5279-improved-multimodal-deep-learning-with-variation-of-information.pdf)]<br />
######for any question, please leave a message: kihyuk.sohn@gmail.com 

##MNIST database

1. move to 'mnist' folder

    `$ cd mnist`

2. open matlab and run scripts.m
    
    `$ matlab`<br />
    `>> optgpu = 1; % 1 for gpu, 0 for cpu`<br />
    `>> gpuDevice(gpu_id); % if optgpu == 1, initialize gpu with proper gpu_id`<br />
    `>> scripts`

    there are 4 versions of training algorithms are implemented:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;a. maximum likelihood (with persistent CD; 'pcd').<br />
    &nbsp;&nbsp;&nbsp;&nbsp;b. CD-percLoss ('cdpl').<br />
    &nbsp;&nbsp;&nbsp;&nbsp;c. multimodal rnn ('mrnn').<br />
    &nbsp;&nbsp;&nbsp;&nbsp;d. hybrid of maximum likelihood and multimodal rnn ('hybrid').<br />


## Flickr database

1. move to 'flickr' folder 

    `$ cd flickr`

2. data preparation

    `$ source prep_data.sh`

    0. download pre-processed features from the following website: http://www.cs.toronto.edu/~nitish/multimodal/index.html<br />
    0. make sure numpy and scipy are installed.

3. get minFunc

    `$ source prep_minFunc.sh`

    0. download unconstrained optimization toolbox from the following website: http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html <br />
    0. we include the minFunc license form at the bottom.

4. open matlab and run scripts.m

    `$ matlab`<br />
    `>> optgpu = 1; % 1 for gpu, 0 for cpu`<br />
    `>> gpuDevice(gpu_id); % if optgpu == 1, initialize gpu with proper gpu_id`<br />
    `>> scripts`<br />

    0. flickr_img_l1[l2]     : image pathway pre-training (2layers)
    0. flickr_text_l1[l2]    : text pathway pretraining (2layers)
    0. flickr_both_l3        : mdrnn training from pretrained weights (3layers)<br/>
    0. we provide pretrained weights in 'results' folder. If you want to train from scratch, you can rename 'results' folder and re-run the scripts.m <br/>
    0. training is composed of 3 steps, such as<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;a. top joint layer pre-training.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;b. fine-tuning top 2 layers.<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;c. fine-tuning the whole network.<br/>




=======================================================================================
The minFunc license is as follows:

The software on this webpage is distributed under the FreeBSD-style license below.

Although it is not required, I would also appreciate that any re-distribution of the
software contains a link to the original webpage.  For example, the webpage for the 
'minFunc' software is: http://www.di.ens.fr/~mschmidt/Software/minFunc.html

Copyright 2005-2012 Mark Schmidt. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=======================================================================================
