cd ..
for dataset in yandex_music_event lastfm1K; do
    echo "testing models with $dataset..."
    python3 two_predictions_significance_test.py --metrics "ndcg@5,ndcg@10,ndcg@40,ndcg@100" --first ./results/predictions/$dataset/BERT4Rec_ppl.json.gz --second ./results/predictions/$dataset/BERT4Rec.json.gz
    python3 two_predictions_significance_test.py --metrics "ndcg@5,ndcg@10,ndcg@40,ndcg@100" --first ./results/predictions/$dataset/SASRec_ppl.json.gz --second ./results/predictions/$dataset/SASRec.json.gz 
    python3 two_predictions_significance_test.py --metrics "ndcg@5,ndcg@10,ndcg@40,ndcg@100" --first ./results/predictions/$dataset/gSASRec_ppl.json.gz --second ./results/predictions/$dataset/gSASRec.json.gz
    echo ""
done;
for dataset in yandex_music_event; do
    echo "testing models with $dataset..."
    python3 two_predictions_significance_test.py --metrics "ndcg@5,ndcg@10" --first ./results/predictions/$dataset/PersonalizedMostPopular.json.gz --second ./results/predictions/$dataset/gSASRec_ppl.json.gz
    python3 two_predictions_significance_test.py --metrics "ndcg@40,ndcg@100" --first ./results/predictions/$dataset/gSASRec_ppl.json.gz  --second ./results/predictions/$dataset/BERT4Rec_ppl.json.gz
    echo ""
done;
for dataset in lastfm1K; do
    echo "testing models with $dataset..."
    python3 two_predictions_significance_test.py --metrics "ndcg@5,ndcg@10,ndcg@40,ndcg@100" --first ./results/predictions/$dataset/PersonalizedMostPopular.json.gz --second ./results/predictions/$dataset/BERT4Rec_ppl.json.gz
    echo ""
done;