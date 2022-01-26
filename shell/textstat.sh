NWORDS=$(wc -w "$1" | cut -f 1 -d " ")
NLINES=$(wc -l "$1" | cut -f 1 -d " ")
WORDSPERLINE=$(($NWORDS / $NLINES))
NUNIQWORDS=$(tr ' ' '\n' < "$1" | sort | uniq | wc -l)

echo "Anzahl Wörter: $NWORDS" | tee "$2"
echo "Anzahl verschiedener Wörter: $NUNIQWORDS" | tee -a "$2"
echo "Durchschnittliche Satzlänge: $WORDSPERLINE" | tee -a "$2"


