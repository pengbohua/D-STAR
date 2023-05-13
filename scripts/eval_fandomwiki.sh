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

python3 eval.py --document-file $documents --model-type bi_encoder --eval-model-path checkpoints/bi_encoder/model_best.ckpt