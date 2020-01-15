# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

HISTFILESIZE=1000000000 HISTSIZE=10000000

bind '"\e[1;5D" backward-word' 
bind '"\e[1;5C" forward-word'
set match-hidden-files off
set completion-ignore-case on
set visible-stats on
set +o noclobber

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

export INPUTRC=$HOME/.inputrc
export FIGNORE="~"
export HISTCONTROL=ignoredups
export HISTCONTROL=ignoreboth
export HISTIGNORE="&:ls:cd *:lh:jobs:bjobs:kill *:[bf]g:top:exit"
export DIR=$HOME
export SOF=$DIR/software
export TMP=$DIR/tmp
#export PATH=$PATH
export PERL5LIB=$PERL5LIB:~/software/perl:/home/vagar/perl5/lib/perl5
export LESS='-S'
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
#export PYTHONPATH=$BASSETDIR/src
#export LUAPATH=$BASSETDIR/src:$LUAPATH

alias sr10='srun --pty --mem 10g /bin/bash -l'
alias sr50='srun --pty --mem 50g /bin/bash -l'
alias sr100='srun --pty --mem 100g /bin/bash -l'
alias grive='google-drive-ocamlfuse ~/google-drive'
alias les='less -N'
alias cdd='cd $DIR'
#alias rm="rm -f"
alias lh="ls -lh"
alias mv="mv -f"
alias gzip='pigz'
alias sq='squeue -u vagar'
alias srgtx='srun --pty --partition=gpu --gres=gpu:gtx1080ti:1 --mem=24g -t 8:00:00 /bin/bash -l'
alias srk80='srun --pty --partition=gpu --gres=gpu:k80:1 --mem=24g -t 8:00:00 /bin/bash -l'
alias srp100='srun --pty --partition=gpu --gres=gpu:p100:1 --mem=24g -t 8:00:00 /bin/bash -l'
alias srrtx='srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=24g -t 8:00:00 /bin/bash -l'
alias si='sinfo -p gpu'
alias hgrep='history | grep '
alias grep='grep --color=always'
alias gitadd="find /home/vagar -maxdepth 1 -name '.bash*' -or -name '.Rprofile' -or -name '.inputrc' | xargs git add; find /home/vagar/predict_expression -name '*.py' -or -name '*.ipynb' -or -name '*.sb' -or -name '*.gin' -or -name '*.joblib' -or -name '*.sh' | xargs git add; git commit -m 'commit'; git push origin master"

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize
shopt -s histappend
shopt -s cdspell
PROMPT_COMMAND="history -a; $PROMPT_COMMAND"

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# Comment in the above and uncomment this below for a color prompt
#PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
PS1='\[\033[01;33m\]\w $ \[\033[00m\]'

# If this is an xterm set the title to user@host:dir
#case "$TERM" in
#xterm*|rxvt*)
#    PROMPT_COMMAND='echo -ne "\033]0;${USER}@${HOSTNAME}: ${PWD/$HOME/~}\007"'
#    ;;
#*)
#    ;;
#esac

# enable color support of ls and also add handy aliases
if [ "$TERM" != "dumb" ]; then
#    eval "$(dircolors -b)"
    alias ls='ls -gGBh --color=auto --format=vertical'
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home/vagar/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#if [ -f "/home/vagar/anaconda3/etc/profile.d/conda.sh" ]; then
#    . "/home/vagar/anaconda3/etc/profile.d/conda.sh"
#    export PATH="/home/vagar/anaconda3/bin:$PATH"
#fi
#fi
#unset __conda_setup
# <<< conda initialize <<<

export BASENJIDIR=~/software/basenji
#export PATH=$BASENJIDIR/bin:~/software:$PATH
export PYTHONPATH=$BASENJIDIR/bin:$BASENJIDIR:/home/vagar/predict_expression/models:$PYTHONPATH
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
source ~/tf1.14/bin/activate
export PATH=$BASENJIDIR/bin:~/software:$PATH:~/software/perl
#grive
