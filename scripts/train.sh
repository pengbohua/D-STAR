#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

if [ -z $DOCDIR]; then
  DOCDIR="data/documents"
fi

if [ -z $MENTIONDIR]; then
  MENTIONDIR="data/Fandomwiki/mentions"
fi

if [ -z $TFIDFDIR]; then
  TFIDFDIR="data/Fandomwiki/tfidf_candidates"
fi

domains=("american_football" "doctor_who" "fallout" "final_fantasy" "military" "pro_wrestling" "starwars" "world_of_warcraft" \
"coronation_street" "elder_scrolls" "ice_hockey" "muppets" "forgotten_realms" "lego" "star_trek" "yugioh")

concat() {
  documents=""
  for domain in $@;
  do documents=${documents:+$documents}$DOCDIR/${domain}.json,;
  done
  documents=${documents::-1}
	echo $documents
}

documents="$(concat ${domains[@]})"

python main.py --document-file $documents \
--train-mentions-file $MENTIONDIR/train.json \
--eval-mentions-file $MENTIONDIR/test.json \
--train-tfidf-candidates-file $TFIDFDIR/train_tfidfs.json \
--eval-tfidf-candidates-file  $TFIDFDIR/test_tfidfs.json \
--train-batch-size 4 \
--eval-batch-size 4 \
--use-tf-idf-negatives \
--max-seq-length 64 \
--learning-rate 1e-5 \
--epochs 2 \
--eval-model-path checkpoint/group_contrastive/