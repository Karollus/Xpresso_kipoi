# Support for perpetual history in bash
#
# NB: obviously be careful storing things forever, as with normal bash
# history any command entered with a space in front will not be recorded


# Setup
#-------------------------------------------------------------------------------

# PERPETUAL_HISTORY_D must be set to history logs directory
if [ -z "$PERPETUAL_HISTORY_D" ]; then
  echo "perpetual history requires PERPETUAL_HISTORY_D env to be set";
  return;
fi

# make PERPETUAL_HISTORY_D if it doesn't exist
if [ ! -d $PERPETUAL_HISTORY_D ]; then
  mkdir $PERPETUAL_HISTORY_D
fi


# Logging hook
#-------------------------------------------------------------------------------

# utility function to trim whitespace at ends of string
# ...bash is a tire fire
trim() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}

# ignore commands beginning with space and duplicate commands
HISTCONTROL=ignoreboth

# global var tracks last command so we only record new commands
perpetual_history__lastcmd=""

# echoes last command-line to log files split by date
perpetual_history::log() {
  local ts=$(date "+%Y-%m-%d.%H:%M:%S")
  local today=$(date +%Y%m%d)
  local pwd=$(pwd)
  # uses fc to print last command from history without cruft
  local cmd=$(fc -lrn | head -1)
  cmd=$(trim $cmd)
  # split cmd string into an array, check command-name
  local cmdname=$(echo $cmd | awk '{print $1;}')
  case "${cmdname}" in           # Ignore commands that are not useful:
    hist|gh|rgh) return ;;       # - ack-grep history
    ls|ll|l|la) return ;;        # - ls and common aliases
    cd) return ;;                # - directory changes
    exit|logout|clear) return ;; # - leaving shell
  esac

  # ensure we aren't root and that we aren't recording repeat commands
  if [[ $(id -u) -ne 0 && $perpetual_history__lastcmd != $cmd ]]; then
    echo "${ts} | ${pwd} | ${cmd}" |
      cut -c1-2000 >> "$PERPETUAL_HISTORY_D/$today.log"
    perpetual_history__lastcmd=${cmd}
  fi
}

# History recording function executes after every command,
# preserve any previous PROMPT_COMMAND
PROMPT_COMMAND+='
perpetual_history::log'


# ack / grep search function
#-------------------------------------------------------------------------------

# primitive argument handling, could be much improved
hist() {
  # if ack is installed use it
  if hash ag 2>/dev/null; then
  ag "$1" "$PERPETUAL_HISTORY_D" | sort
else
  grep "$1" "$PERPETUAL_HISTORY_D"/* | sort
echo "consider installing ag: sudo apt-get install silversearcher-ag"
  fi
}

