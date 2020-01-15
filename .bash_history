head -2 dev.tsv
tail -2 train.tsv
grep -A2 YOL136C yeast_all_data.shuffled.tsv
tail -n+2401 yeast_all_data.shuffled.tsv | head -86 | cat <(head -1 yeast_all_data.tsv) - >test.tsv
tail -2 dev.tsv 
grep -A2 YJL079C yeast_all_data.shuffled.tsv
head -2 test.tsv 
tail -2 test.tsv 
tail -1 yeast_all_data.shuffled.tsv
tail -n+2401 yeast_all_data.shuffled.tsv | head -87 | cat <(head -1 yeast_all_data.tsv) - >test.tsv
tail -2 test.tsv 
less test.tsv
tail -n+2401 yeast_all_data.shuffled.tsv | head -88 | cat <(head -1 yeast_all_data.tsv) - >test.tsv
grep YMR296C test.tsv 
grep YMR296C yeast_all_data.shuffled.tsv
les yeast_all_data.shuffled.tsv
tail -n+2401 yeast_all_data.shuffled.tsv | head -187 | cat <(head -1 yeast_all_data.tsv) - >test.tsv
tail -2 test.tsv 
head -2200 yeast_all_data.shuffled.tsv | cat <(head -1 yeast_all_data.tsv) - >train.tsv
tail -n+2201 yeast_all_data.shuffled.tsv | head -200 | cat <(head -1 yeast_all_data.tsv) - >dev.tsv
wc -l *tsv
calc 2200+200+185
tail -1 *tsv
tail -n1 *tsv
grep -A2 YML108W yeast_all_data.shuffled.tsv
head -2 *.tsv 
grep -A2 YJL079C yeast_all_data.shuffled.tsv
python
python run_regression.py --help
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000.index 
calc 93*2
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=50g -t 8:00:00 /bin/bash -l
lh yeast3utrFineTuned/
tensorboard yeast3utrFineTuned/
tensorboard --logdir yeast3utrFineTuned --host localhost --port 2223
lh yeast3utrFineTuned/
tensorboard --logdir yeast3utrFineTuned2 --host localhost --port 2223
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000
nvidia-smi 
lh yeast3utrFineTuned/
ls yeast3utrFineTuned/
less yeast3utrFineTuned/eval_results.txt 
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000
lh yeast3utrFineTuned/
ls yeast3utrFineTuned/
less yeast3utrFineTuned/test_results.tsv 
less yeast3utrFineTuned/eval_results.tsv 
less yeast3utrFineTuned/eval_results.txt 
wc -l yeast3utrFineTuned/test_results.tsv 
less yeast3utrFineTuned/submit_results.tsv 
R
python run_regression.py --help
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000 
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000 --train_step 10000 --eval_batch_size 32
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned2 --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000 --train_step 10000 --eval_batch_size 32
python run_regression.py --albert_config_file albert_config.json --data_dir yeast3UTR --do_eval --do_lower_case --do_predict --do_train --output_dir yeast3utrFineTuned2 --spm_model_file yeast3UTR/yeast.model --task_name yeast --vocab_file yeast3UTR/yeast.stripped.vocab --init_checkpoint yeast3utrModel/model.ckpt-10000 --train_step 1000 --eval_batch_size 32
R
python run_regression.py --help
R
python run_regression.py --help
less yeast3utrFineTuned2/test_results.tsv
python
n
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=20g -t 8:00:00 /bin/bash -l
lh yeast3UTR*
lh yeast3utrFineTuned*
lh yeast3UTR
less ../../yeastorfs.fa 
unjustify_fasta.pl yeastorfs.fa 
mv yeastorfs.fa.prepared yeastorfs.fa
less yeastorfs.fa 
hgrep yeast3utr.lengths.txt
hgrep yeast3utr.txt
awk 'NR % 2==0 {print}' yeastorfs.fa >yeastorfs.txt
less yeastorfs.txt 
awk '{print length($0);}' yeastorfs.txt >yeastorfs.lengths.txt
R
spm_train --input=yeastorfs.txt --model_prefix=yeastORF --vocab_size=5 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --model_type=char
cat yeastORF.vocab yeast.vocab
cat yeast.stripped.vocab
cat yeastORF.vocab
less yeastORF.model
spm_encode --model=yeastORF.model --output_format=piece <yeastorfs.txt | perl -ne 'print substr($_,4)."\n";' >yeastorfs.encoded.txt 
less yeastorfs.encoded.txt 
spm_encode --model=yeastORF.model --output_format=piece <yeastorfs.txt | less
less yeastorfs.encoded.txt 
spm_encode --model=yeastORF.model --output_format=piece <yeastorfs.txt | less
less yeastorfs.encoded.txt 
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=50g -t 8:00:00 /bin/bash -l
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=40g -t 8:00:00 /bin/bash -l
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=30g -t 8:00:00 /bin/bash -l
time python create_pretraining_data.py --spm_model_file=yeastTest/yeastORF.model --input_file=yeastTest/yeastorfs.encoded.txt --output_file=yeastTest/yeastorfsWithSPM.tfrecord --short_seq_prob=0.8 --vocab_file=yeastTest/yeast.stripped.vocab --max_predictions_per_seq=50 --dupe_factor=100 --max_seq_length=2048 &
. ~/.bashrc 
mv yeast3UTR yeastTest
time python create_pretraining_data.py --spm_model_file=yeastTest/yeastORF.model --input_file=yeastTest/yeastorfs.encoded.txt --output_file=yeastTest/yeastorfsWithSPM.tfrecord --short_seq_prob=0.8 --vocab_file=yeastTest/yeast.stripped.vocab --max_predictions_per_seq=50 --dupe_factor=100 --max_seq_length=2048 &
rm yeastTest/yeastorfsWithSPM.tfrecord 
echo $HOSTNAME
time python create_pretraining_data.py --spm_model_file=yeastTest/yeastORF.model --input_file=yeastTest/yeastorfs.encoded.txt --output_file=yeastTest/yeastorfsWithSPM.tfrecord --short_seq_prob=0.8 --vocab_file=yeastTest/yeast.stripped.vocab --max_predictions_per_seq=100 --dupe_factor=100 --max_seq_length=2048
wc -l yeastorfs.txt
less yeastorfs.
less yeastorfs.txt 
leass yeastorfs.fa 
less yeastorfs.fa 
awk 'NR % 2==1 {print}' yeastorfs.fa | perl -ne '($id, $chr, $start, $stop, $str) = ($_ =~ /(Y.*)\_SC.*(chr\w+):(\d+)-(\d+).*strand=(.)/); print join("\t", $chr, $start, $stop, $id,".",$str)."\n"; ' >yeastorfs.bed
less yeastorfs.
less yeastorfs.bed 
awk 'NR % 2==1 {print}' <yeastorfs.fa | perl -ne '($id, $chr, $start, $stop, $str) = ($_ =~ /(Y.*)\_SC.*(chr\w+):(\d+)-(\d+).*strand=(.)/); print join("\t", $chr, $start, $stop, $id,".",$str)."\n"; ' >yeastorfs.bed
less yeastorfs.bed 
less yeastorfs.fa 
awk 'NR % 2==1 {print}' <yeastorfs.fa | perl -ne '($id, $chr, $start, $stop, $str) = ($_ =~ /(Y.*)\ .*(chr\w+):(\d+)-(\d+).*strand=(.)/); print join("\t", $chr, $start, $stop, $id,".",$str)."\n"; ' >yeastorfs.bed
less yeastorfs.fa 
less yeastorfs.bed 
sort -k4,4 yeastorfs.bed | less
less yeastorfs.bed 
less yeastorfs.fa 
awk 'NR % 2==1 {print}' <yeastorfs.fa | perl -ne '($id, $chr, $start, $stop, $str) = ($_ =~ /(Y|Q.*)\ .*(chr\w+):(\d+)-(\d+).*strand=(.)/); print join("\t", $chr, $start, $stop, $id,".",$str)."\n"; ' >yeastorfs.bed
less yeastorfs.fa 
less yeastorfs.bed 
awk 'NR % 2==1 {print}' <yeastorfs.fa | perl -ne '($id, $chr, $start, $stop, $str) = ($_ =~ /([Y|Q].*)\ .*(chr\w+):(\d+)-(\d+).*strand=(.)/); print join("\t", $chr, $start, $stop, $id,".",$str)."\n"; ' >yeastorfs.bed
less yeastorfs.bed 
sort -k4,4 yeastorfs.bed | less
wc -l yeastorfs.bed 
echo $HOSTNAME
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq 100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=2048 
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq 100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length 2048 
python run_pretraining.py --output_dir yeastORFModel2 --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq 100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=2048 
python run_pretraining.py --output_dir yeastORFModel2 --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq 100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=512
python run_pretraining.py --output_dir yeastORFModel2 --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq 50 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=512
python run_pretraining.py --output_dir yeastORFModel2 --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=512
python run_pretraining.py --output_dir yeastORFModel3 --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=512
rm -r yeastORFModel yeastORFModel2 yeastORFModel3
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=2048
sudo systemctl restart nxserver
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=20g -t 8:00:00 /bin/bash -l
tensorboard --logdir yeastORFModel --host localhost --port 2223
b g
rm pnas.1817299116.sd05\ \(1\).tsv 
less pnas.1817299116.sd01.tsv 
wc -l pnas.1817299116.sd01.tsv 
wc -l pnas.1817299116.sd02.tsv 
less pnas.1817299116.sd02.tsv 
rm pnas.1817299116.sd03.xlsx 
less pnas.1817299116.sd04.tsv 
wc -l pnas.1817299116.sd04.tsv 
rm pnas.1817299116.sd05.tsv 
rm pnas.1817299116.sd06.tsv 
less pnas.1817299116.sd07.tsv 
rm pnas.1817299116.sd07.tsv 
less pnas.1817299116.sd08.tsv 
rm pnas.1817299116.sd08.tsv 
mkdir Riba_Zavolan_2019
mv pnas.1817299116.sd0* Riba_Zavolan_2019/
cat >README
cat README 
less Riba_Zavolan_2019/pnas.1817299116.sd0
head -1 Riba_Zavolan_2019/pnas.1817299116.sd01.tsv 
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=32 --eval_batch_size=32 --max_seq_length=2048
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=8 --eval_batch_size=32 --max_seq_length=2048
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=16 --eval_batch_size=32 --max_seq_length=2048
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=16 --eval_batch_size=16 --max_seq_length=2048
python run_pretraining.py --output_dir yeastORFModel --do_train --do_eval --input_file yeastTest/yeastorfsWithSPM.tfrecord --albert_config_file albert_config.json --max_predictions_per_seq=100 --train_batch_size=8 --eval_batch_size=128 --max_seq_length=2048
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=50g -t 8:00:00 /bin/bash -l
ls yeastORFModel/
lh yeastTest
hgrep nexu
grep ssh ~/.bashrc 
less ~/.bashrc 
sshfs vagar@nexus.gs.washington.edu:/net/shendure/ /home/vagar/UW2
less yeastorfs.txt 
hgrep yeastorfs.txt 
hgrep yeastorfs.txt
less yeastorfs.lengths.txt
perl -ne 'print $_/3;' < yeastorfs.lengths.txt | less
perl -ne 'print $_/3."\n";' < yeastorfs.lengths.txt | less
perl -ne 'print ($_/3)."\n";' < yeastorfs.lengths.txt | less
perl -ne 'chomp; print ($_/3)."\n";' < yeastorfs.lengths.txt | less
perl -ne 'chomp; print $_/3; print "\n";' < yeastorfs.lengths.txt | less
perl 'chomp; for ($i=0; $i < length($i); $i+=3){ print $_[$i..($i+3)]." "; } print "\n";' <yeastorfs.txt | less
perl 'chomp; for($i=0; $i < length($i); $i+=3){ print substr($_,$i,$i+3)." "; } print "\n";' <yeastorfs.txt | less
perl 'chomp; for($i=0; $i < length($_); $i+=3){ print substr($_,$i,$i+3)." "; } print "\n";' <yeastorfs.txt | less
perl 'chomp; for($i=0; $i < length($_); $i+=3; ){ print substr($_,$i,$i+3)." "; } print "\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_); $i+=3; ){ print substr($_,$i,$i+3)." "; } print "\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3; ){ print substr($_,$i,3)." "; } print "\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print "\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_); $i+=3){ print substr($_,$i,3)." "; } print "\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_); $i+=3){ print substr($_,$i,3)." "; } print "zz\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."zz\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n\n";' <yeastorfs.txt | less
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n\n";' <yeastorfs.txt >yeastorfs.codons.encoded.txt 
less yeastorfs.encoded.txt
less yeastorfs.codons.encoded.txt
less yeastorfs.txt 
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n\n";' <yeastorfs.txt >yeastorfs.codons.txt 
less yeastorfs.codons.txt 
perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n";' <yeastorfs.txt >yeastorfs.codons.txt 
less yeastorfs.codons.txt 
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=68 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=100 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=68 -bos_id=0--eos_id=-1 --unk_id=4 --character_coverage=1
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon
less yeastorfs.codons.txt
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=6000
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=8000
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=10000
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --input_sentence_size=7000
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --input_sentence_size=6692
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=64 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=63 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=65 -bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --hard_vocab_limit=false
less yeastORFwCodon.vocab
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=65 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --hard_vocab_limit=false
cat yeastORF.vocab
cat yeastORFwCodon.vocab 
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=65 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --hard_vocab_limit=false --model_type=char
cat yeastORFwCodon.vocab 
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=65 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --hard_vocab_limit=false --model_type=word
cat yeastORFwCodon.vocab 
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 --bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --hard_vocab_limit=false --model_type=word
less yeastORFwCodon.vocab
les yeastORFwCodon.vocab
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 --bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --model_type=word
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=70 --bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --model_type=word
spm_train --input=yeastorfs.codons.txt --model_prefix=yeastORFwCodon --vocab_size=66 --bos_id=-1 --eos_id=-1 --unk_id=4 --character_coverage=1 --max_sentence_length=20000 --hard_vocab_limit=false --model_type=word
ssh nexus.gs.washington.edu
hgrep vagar
ssh vagar@nexus.gs.washington.edu
ssh vagar@nexus.gs.washington.edu
mkdir Weinberg_2016
less weinberg_synthesis.txt 
R
less dev.tsv 
hgrep test.tsv
mv yeast_all_data.* test.tsv train.tsv dev.tsv Manuscript_Cheng_RNA_2017/
mkdir yeast3utr
mv yeast3utr* yeast3utr/
mkdir yeast.* yeast3utr/
mv yeast.* yeast3utr/
mkdir yeastorfs
mv yeastorfs* yeastorfs/
mv yeastORF* yeastorfs/
less SGD_all_ORFs_3prime_UTRs.bed
less SGD_all_ORFs_3prime_UTRs.fsa
less SGD_all_ORFs_3prime_UTRs.uniq.bed
mv SGD_all_ORFs_3prime_UTRs.* Manuscript_Cheng_RNA_2017/
less README 
cat README 
mv README Manuscript_Cheng_RNA_2017/
ls ../
less ../sentencepiece/sentencepiece.pc.in 
ls ../../
ls Riba_Zavolan_2019/
ls Riba_Zavolan_2019/pnas.1817299116.sd01.tsv 
less Riba_Zavolan_2019/pnas.1817299116.sd01.tsv 
rm -r Riba_Zavolan_2019/
ls Manuscript_Cheng_RNA_2017/
mkdir half_life
mv dev.tsv SGD_all_ORFs_3prime_UTRs.* yeast_all_data.* train.tsv test.tsv ../half_life/
less README
mv README ../half_life/
ls half_life/
less half_life/dev.tsv 
l
mkdir with3utr
ls dev.tsv train.tsv test.tsv with3utr/
mv dev.tsv train.tsv test.tsv with3utr/
mkdir withORF
less yeast_all_data.shuffled.tsv 
ls ../
less SGD_all_ORFs_3prime_UTRs.bed 
lh ../yeast3utr/
mv SGD_all_ORFs_3prime_UTRs.* ../yeast3utr/
less yeast_all_data.tsv 
wc -l yeast_all_data.tsv 
hgrep yeast_all_data.tsv 
less Manuscript_Cheng_RNA_2017/.Rhistory 
head -2 yeast_all_data.tsv 
less yeast_all_data.shuffled.tsv 
ls with3utr/
mv with3utr/* ../yeast3utr/
rmdir with3utr/
mv yeast_all_data.* README ../yeast3utr/
ls withORF/
rmdir withORF/
rm -r half_life/
ls yeastorfs/
ls yeast3utr/
ls yeastorfs/
less ../yeast3utr/train.tsv 
less yeastorfs.codons.encoded.txt 
less yeastorfs.codons.txt 
less yeastorfs.encoded.txt 
less yeastorfs.txt 
less yeastorfs.fa 
hgrep yeastorfs.fa
hgrep perl | tail -20
less yeastorfs.bed 
cut -f 4 yeastorfs.bed | less
cut -f 4 yeastorfs.bed | paste - yeastorfs.txt | less
cut -f 4 yeastorfs.bed | paste - yeastorfs.txt >yeastorfs.withIDs.txt 
less yeastorfs.withIDs.txt 
ln -s ../yeast3utr/dev.tsv 
ln -s ../yeast3utr/test.tsv 
ln -s ../yeast3utr/train.tsv 
less dev.tsv 
perl -ne '@a=split /\t/; print $a[0],"\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/; print $a[0],"\n"; $line=`grep $a[0] yeastorfs.withIDs.txt`; print $line;' <dev.tsv | head -2
perl -ne '@a=split /\t/; print $a[0],"\n"; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print $line;' <dev.tsv | head -2
cut -f 4 yeastorfs.bed | paste - yeastorfs.codons.txt >yeastorfs.withIDs.txt 
less yeastorfs.withIDs.txt
perl -ne '@a=split /\t/; print $a[0],"\n"; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print $line;' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, @a[2..-1])."\n"; if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $@a[2..-1])."\n"; if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $a[2..-1])."\n"; if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $a[2..$#a])."\n"; if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $a[2..$#a])."\n" if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $a[1..$#a])."\n" if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, @a[1..$#a])."\n" if $line eq "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, @a[1..$#a])."\n" if $line ne "";' <dev.tsv | head -2
less dev.tsv 
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, @a[1..$#a])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $a[1..$#a])."\n" if $line ne "";' <dev.tsv | head -1
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, \@a[1..$#a])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; print join("\t", $a[0], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[1])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -4
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep \$a[0] yeastorfs.withIDs.txt`; chomp $line; print $); join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $); join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "^\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "^\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -3
perl -ne '@a=split /\t/; $line=`grep -P "\^\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -3
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -3
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -3
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
less dev.tsv 
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\\t" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\\\t" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print $a[0]."\t".$line; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; join("\t", $a[0], $a[2], $line, $a[2])."\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/; join("\t", $a[0], $a[2], $a[2])."\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/; join("\t", $a[0], $a[2], $a[2])."\n";' <dev.tsv 
less dev.tsv 
perl -ne '@a=split /\t/, $_; join("\t", $a[0], $a[2], $a[2])."\n";' <dev.tsv 
perl -ne '@a=split /\t/, $_; join("\t", $a[0], $a[2], $a[2])."\n";' <dev.tsv | head -2
which perl
perl -ne '@a=split /\t/, $_; join("\t", $a[0], $a[1], $a[2])."\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/, $_; join("\t", $a[0])."\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/, $_; print join("\t", $a[0])."\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]\t" yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[2], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[1], $line, $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[1], $a[2])."\n" if $line ne "";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[1], $line, $a[2])."\n";' <dev.tsv | head -2
perl -ne '@a=split /\t/; $line=`grep -P "\$a[0]" yeastorfs.withIDs.txt`; chomp $line; print join("\t", $a[0], $a[1], $line, $a[2])."\n";' <dev.tsv | head -1
touch gen_training_set.pl
less yeastorfs.withIDs.txt 
chmod +x gen_training_set.pl 
./gen_training_set.pl dev.tsv | less
rm dev.tsv train.tsv test.tsv 
./gen_training_set.pl ../yeast3utr/dev.tsv | less
./gen_training_set.pl ../yeast3utr/dev.tsv >dev.tsv
less dev.tsv 
./gen_training_set.pl ../yeast3utr/dev.tsv >dev.tsv
b
./gen_training_set.pl ../yeast3utr/train.tsv >train.tsv &
./gen_training_set.pl ../yeast3utr/test.tsv >test.tsv &
wc -l *tsv
wc -l ../yeast3utr/*tsv
less train.tsv 
nano train.tsv 
nano test.tsv 
nano dev.tsv 
less albert_config.json 
ls yeastORFModel/
hgrep yeastorfsWithSPM.tfrecord
less yeastorfs.codons.encoded.txt
less yeastorfs.codons.txt
less yeastORFwCodon.vocab
les yeastORFwCodon.vocab
cat *vocab
ls *vocab
ls ../Manuscript_Cheng_RNA_2017/
ls ../yeast3utr/
cat ../yeast3utr/yeast.stripped.vocab 
cut -f 1 yeastORFwCodon.vocab 
cut -f 1 yeastORFwCodon.vocab >yeastORFwCodon.stripped.vocab
cat ../yeast3utr/yeast.stripped.vocab 
less yeastorfs.encoded.txt 
less yeastorfs.codons.encoded.txt 
time python create_pretraining_data.py --spm_model_file=yeastTest/yeastorfs/yeastORFwCodon.model --input_file=yeastTest/yeastorfs/yeastorfs.codons.encoded.txt --output_file=yeastTest/yeastorfs/yeastorfsWithSPM.codons.tfrecord --short_seq_prob=0.8 --vocab_file=yeastTest/yeastorfs/yeastORFwCodon.stripped.vocab --max_predictions_per_seq=50 --dupe_factor=100 --max_seq_length=1024
ls
sudo systemctl restart nxserver
cp albert_config.json albert_config_codons.json 
python run_pretraining.py --output_dir yeastORFModelwCodons --do_train --do_eval --input_file yeastTest/yeastorfs/yeastorfsWithSPM.codons.tfrecord --albert_config_file albert_config_codons.json --max_predictions_per_seq=100 --train_batch_size=32 --eval_batch_size=128 --max_seq_length=1024
python run_pretraining.py --output_dir yeastORFModelwCodons --do_train --do_eval --input_file yeastTest/yeastorfs/yeastorfsWithSPM.codons.tfrecord --albert_config_file albert_config_codons.json --max_predictions_per_seq=50 --train_batch_size=32 --eval_batch_size=128 --max_seq_length=1024
python run_pretraining.py --output_dir yeastORFModelwCodons --do_train --do_eval --input_file yeastTest/yeastorfs/yeastorfsWithSPM.codons.tfrecord --albert_config_file albert_config_codons.json --max_predictions_per_seq=50 --train_batch_size=16 --eval_batch_size=128 --max_seq_length=1024
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=50g -t 8:00:00 /bin/bash -l
tensorboard --logdir yeastORFModelwCodons --host localhost --port 2223
less yeastTest/yeastorfs/dev.tsv 
python run_regression.py --albert_config_file albert_config_codons.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfswCodonFineTuned --spm_model_file yeastTest/yeastorfs/yeastORFwCodon.model --task_name yeast --vocab_file yeastTest/yeastorfs/yeastORFwCodon.stripped.vocab --init_checkpoint yeastORFModelwCodons/model.ckpt-10000 --train_step 1000 --eval_batch_size 32
tensorboard --logdir yeastOrfswCodonFineTuned --host localhost --port 2223
ls yeastORFModelwCodons/
ls yeastOrfswCodonFineTuned/
R
python run_regression.py --albert_config_file albert_config_codons.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfswCodonFineTuned --spm_model_file yeastTest/yeastorfs/yeastORFwCodon.model --task_name yeast --vocab_file yeastTest/yeastorfs/yeastORFwCodon.stripped.vocab --init_checkpoint yeastORFModelwCodons/model.ckpt-10000 --train_step 2000 --eval_batch_size 32
R
python run_regression.py --albert_config_file albert_config.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfsFineTuned --spm_model_file yeastTest/yeastorfs/yeastORF.model --task_name yeast --vocab_file yeastTest/yeast3utr/yeast.stripped.vocab --init_checkpoint yeastORFModelwCodonscccccckieveldneigrvfktejccthfgrthrrfdukbihe/model.ckpt-10000 --train_step 2000 --eval_batch_size 32
python run_regression.py --albert_config_file albert_config.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfsFineTuned --spm_model_file yeastTest/yeastorfs/yeastORF.model --task_name yeast --vocab_file yeastTest/yeast3utr/yeast.stripped.vocab --init_checkpoint yeastORFModel/model.ckpt-10000 --train_step 2000 --eval_batch_size 32
python
python run_regression.py --albert_config_file albert_config.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfsFineTuned --spm_model_file yeastTest/yeastorfs/yeastORF.model --task_name yeast --vocab_file yeastTest/yeast3utr/yeast.stripped.vocab --init_checkpoint yeastORFModel/model.ckpt-10000 --train_step 1000 --eval_batch_size 32
python run_regression.py --albert_config_file albert_config.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfsFineTuned --spm_model_file yeastTest/yeastorfs/yeastORF.model --task_name yeast --vocab_file yeastTest/yeast3utr/yeast.stripped.vocab --init_checkpoint yeastORFModel/model.ckpt-10000 --train_step 2000 --eval_batch_size 32
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=50g -t 8:00:00 /bin/bash -l
ls glue/
tensorboard --logdir yeastOrfsFineTuned --host localhost --port 2223
R
less yeastORFwCodon.vocab 
less yeastORFwCodon.stripped.vocab 
cp gen_training_set.pl gen_lm_set.pl 
./gen_lm_set.pl 
perl gen_lm_set.pl 
perl gen_lm_set.pl <train.tsv 
perl gen_lm_set.pl <train.tsv | less
perl gen_lm_set.pl <train.tsv >train.forLM.tsv 
perl gen_lm_set.pl <test.tsv >test.forLM.tsv 
perl gen_lm_set.pl <dev.tsv >dev.forLM.tsv 
R
less residuals.txt 
less train.residuals.tsv 
wc -l *tsv
less train.residuals.tsv 
sort -k1,1 dev.residuals.tsv | uniq -c | sort -nr | less
sort -k1,1 <(cut -f 1 dev.residuals.tsv) | uniq -c | sort -nr | less
less train.residuals.tsv 
sort -k1,1 <(cut -f 1 train.residuals.tsv) | uniq -c | sort -nr | less
less train.residuals.tsv 
uniq train.residuals.tsv | wc -l
less train.residuals.tsv 
sort -k1,1 <(cut -f 1 train.residuals.tsv) | uniq -c | sort -nr | less
less train.residuals.tsv 
nano train.residuals.tsv 
sort -k1,1 <(cut -f 1 train.residuals.tsv) | uniq -c | sort -nr | less
nano train.residuals.tsv 
sort -k1,1 <(cut -f 1 train.residuals.tsv) | uniq -c | sort -nr | less
nano train.residuals.tsv 
sort -k1,1 <(cut -f 1 train.residuals.tsv) | uniq -c | sort -nr | less
nano train.residuals.tsv 
sort -k1,1 <(cut -f 1 dev.residuals.tsv) | uniq -c | sort -nr | less
nano dev.residuals.tsv 
sort -k1,1 <(cut -f 1 test.residuals.tsv) | uniq -c | sort -nr | less
wc -l *tsv
less train.residuals.tsv 
python run_regression.py --albert_config_file albert_config_codons.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfswCodonResidualsFineTuned --spm_model_file yeastTest/yeastorfs/yeastORFwCodon.model --task_name yeast --vocab_file yeastTest/yeastorfs/yeastORFwCodon.stripped.vocab --init_checkpoint yeastORFModelwCodons/model.ckpt-10000 --train_step 1000 --eval_batch_size 32
python run_regression.py --albert_config_file albert_config_codons.json --data_dir yeastTest/yeastorfs --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfswCodonResidualsFineTuned --spm_model_file yeastTest/yeastorfs/yeastORFwCodon.model --task_name yeast --vocab_file yeastTest/yeastorfs/yeastORFwCodon.stripped.vocab --init_checkpoint yeastORFModelwCodons/model.ckpt-10000 --train_step 2000 --eval_batch_size 32
srun --pty --partition=gpu24 --gres=gpu:titanrtx:1 --mem=50g -t 8:00:00 /bin/bash -l
tensorboard --logdir yeastOrfswCodonResidualsFineTuned --host localhost --port 2223
ls yeastorfs/
ls Weinberg_2016/
less Weinberg_2016/weinberg_synthesis.txt 
R
less initiation_rates.txt 
wc -l initiation_rates.txt 
mkdir yeasttranslation
rmdir yeasttranslation/
less ../yeastorfs/yeastorfs.withIDs.txt 
wc -l ../yeastorfs/yeastorfs.withIDs.txt 
ln -s ../yeastorfs/yeastorfs.withIDs.txt 
less weinberg_synthesis.txt 
R
less initiation_rates_withORF.txt 
nano initiation_rates_withORF.txt 
shuf initiation_rates_withORF.txt >initiation_rates_withORF_shuffled.txt 
wc -l *txt
less initiation_rates_withORF_shuffled.txt 
head -3000 initiation_rates_withORF_shuffled.txt >train.tsv
tail -n+3001 initiation_rates_withORF_shuffled.txt | head -1000 >dev.tsv
tail -n+4001 initiation_rates_withORF_shuffled.txt | head -839 >dev.tsv
tail -n+3001 initiation_rates_withORF_shuffled.txt | head -1000 >dev.tsv
tail -n+4001 initiation_rates_withORF_shuffled.txt | head -839 >test.tsv
wc -l *tsv
tail -1 train.tsv | cut -f 1
head -1 dev.tsv | cut -f 1
less initiation_rates_withORF_shuffled.txt
tail -1 dev.tsv | cut -f 1
head -1 test.tsv | cut -f 1
less initiation_rates_withORF_shuffled.txt
python run_regression.py --albert_config_file albert_config_codons.json --data_dir yeastTest/Weinberg_2016/ --do_eval --do_lower_case --do_predict --do_train --output_dir yeastOrfswCodonResiduals_TranslationRate_FineTuned --spm_model_file yeastTest/yeastorfs/yeastORFwCodon.model --task_name yeast --vocab_file yeastTest/yeastorfs/yeastORFwCodon.stripped.vocab --init_checkpoint yeastORFModelwCodons/model.ckpt-10000 --train_step 1000 --eval_batch_size 32
R
tensorboard --logdir yeastOrfswCodonResiduals_TranslationRate_FineTuned/ --host localhost --port 2223
wc -l yeastTest/Weinberg_2016/*tsv
wget https://downloads.yeastgenome.org/sequence/fungi/L_kluyveri/NRRL-Y12651/NRRL-Y12651_Utah_2010_AACE03000000.fsa.gz
less NRRL-Y12651_Utah_2010_AACE03000000.fsa.gz 
rm NRRL-Y12651_Utah_2010_AACE03000000.fsa.gz 
df -h
mkdir resources
mkdir yeast
ftp hgdownload.cse.ucsc.edu 
mkdir maf
mkdir multizi7way
mv multizi7way/ multiz7way
mkdir multiz7way/maf
mv *gz multiz7way/maf/
rm *md5
sl
rm md5sum.txt 
lh maf/
rmdir maf
ls multiz7way/
mv 7way.nh multiz7way/
rm upstream*
s
ssh vagar@nexus.gs.washington.edu
sshfs vagar@nexus.gs.washington.edu:/net/shendure/ /home/vagar/UW2
cp ~/UW2/vol1/home/vagar/software/* .
cp -r ~/UW2/vol1/home/vagar/software/bash_scripts/ .
cp -r ~/UW2/vol1/home/vagar/software/perl/ .
ls perl/
mafFrag
mafSplit
sudo dpkg -i libpng12-0_1.2.54-1ubuntu1.1_amd64.deb
uname -a
ldconfig | grep libpng
ldconfig 
sudo ldconfig 
ldconfig
sudo ldconfig
less /etc/ld.so.cache 
wget http://security.ubuntu.com/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1.1_amd64.deb
sudo dpkg -i libpng12-0_1.2.54-1ubuntu1.1_amd64.deb
rm libpng12-0_1.2.54-1ubuntu1.1_amd64.deb 
mafSplit
mafFrag
find ~/UW2/vol1/home/vagar/projects/ -name '*py'
find ~/UW2/vol1/home/vagar/projects/ -name '*maf'
find ~/UW2/vol1/home/vagar/projects/ -name '*maf*'
less ~/software/perl/maf2clustal.pl 
find ~/UW2/vol1/home/vagar/projects/promoter_prediction/ -name '*maf*'
find ~/UW2/vol1/home/vagar/projects/expression_prediction/ -name '*maf*'
find ~/UW2/vol10/projects/expression_prediction/ -name '*maf*'
ssh vagar@nexus.gs.washington.edu
mafFrag
mafFrags
mafFrag
mafFetch 
mafFrag hg19 multiz100way 1 100 200
mafFrag hg19 multiz100way 1 100 200 test.maf
mafFrag hg19 multiz100way 1 100 200 + test.maf
less ~/.hg.conf
nano ~/.hg.conf
ls ~
echo ~
mafFrag hg19 multiz7way 1 100 200 + test.maf
lh ~/.hg.conf 
chmod u+rx ~/.hg.conf 
lh ~/.hg.conf 
chmod -rx ~/.hg.conf 
lh ~/.hg.conf 
chmod u+rx ~/.hg.conf 
lh ~/.hg.conf 
mafFrag hg19 multiz7way 1 100 200 + test.maf
mysql -h genome-mysql.soe.ucsc.edu -ugenome -A -e "select * from ncbiRefSeq limit 2" hg38
sudo apt-get install mysql-client
mysql -h genome-mysql.soe.ucsc.edu -ugenome -A -e "select * from ncbiRefSeq limit 2" hg38
mafFetch xenTro9 multiz11way region.bed stdout
mafFrag hg19 multiz7way 1 100 200 + test.maf
rm ~/.hg.conf 
mafFrag hg19 multiz7way 1 100 200 + test.maf
nano ~/.hg.conf
mafFrag hg19 multiz7way 1 100 200 + test.maf
chmod -rx ~/.hg.conf 
chmod u+rx ~/.hg.conf 
mafFrag hg19 multiz7way 1 100 200 + test.maf
mafFrag hg19 multiz7way chr1 100 200 + test.maf
mafFrag hg19 multiz100way chr1 100 200 + test.maf
less test.maf 
rm test.maf 
cat 7way.nh 
mafFrag sacCer3 multiz7way chrI 100 200 + test.maf
cat test.maf 
rm -r maf/
ls test.maf 
less test.maf 
less ~/software/perl/maf2clustal.pl 
nano ~/.bashrc
. ~/.bashrc 
maf2clustal.pl 
cpanm Bio::AlignIO
sudo apt-get install cpanminus
cpanm Bio::AlignIO
maf2clustal.pl 
nano ~/.bashrc
. ~/.bashrc 
maf2clustal.pl 
cpanm Bio::AlignIO
less /home/vagar/.cpanm/work/1579037428.2564/build.log
perl -e 'use Bio::AlignIO'
ls ~/perl5
cpanm XML::DOM::XPath
wget https://cpan.metacpan.org/authors/id/M/MI/MIROD/XML-DOM-XPath-0.14.tar.gz
tar -xvzf XML-DOM-XPath-0.14.tar.gz
nano XML-DOM-XPath-0.14/t/test_non_ascii.t
tar -czvf XML-DOM-XPath-0.14.tar.gz XML-DOM-XPath-0.14/
cpanm XML-DOM-XPath-0.14
cpanm XML-DOM-XPath-0.14/
cpanm XML-DOM-XPath-0.14.tar.gz 
nano XML-DOM-XPath-0.14/t/test_non_ascii.t
tar -czvf XML-DOM-XPath-0.14.tar.gz XML-DOM-XPath-0.14/
nano XML-DOM-XPath-0.14/t/test_non_ascii.t
tar -czvf XML-DOM-XPath-0.14.tar.gz XML-DOM-XPath-0.14/
cpanm XML-DOM-XPath-0.14.tar.gz 
cpanm Bio::AlignIO
perl -e 'use Bio::AlignIO'
nano ~/.bashrc 
. ~/.bashrc 
perl -e 'use Bio::AlignIO'
rm -r multiz7way/
rm -r resources/
less SGD_all_ORFs_3prime_UTRs.bed 
less SGD_all_ORFs_3prime_UTRs.uniq.bed 
hgrep SGD_all_ORFs_3prime_UTRs.uniq.bed
less SGD_all_ORFs_3prime_UTRs.uniq.bed 
wc -l SGD_all_ORFs_3prime_UTRs.uniq.bed 
less SGD_all_ORFs_3prime_UTRs.uniq.bed 
mafFrags sacCer3 multiz7way SGD_all_ORFs_3prime_UTRs.uniq.bed test.maf
mafFrags
less SGD_all_ORFs_3prime_UTRs.
less SGD_all_ORFs_3prime_UTRs.uniq.bed 
perl -ne '@a=split; $a[5]=0; print join("\t", @a);' <SGD_all_ORFs_3prime_UTRs.uniq.bed | less
perl -ne '@a=split; $a[4]=0; print join("\t", @a);' <SGD_all_ORFs_3prime_UTRs.uniq.bed | less
perl -ne '@a=split; $a[4]=0; print join("\t", @a), "\n";' <SGD_all_ORFs_3prime_UTRs.uniq.bed | less
perl -ne '@a=split; $a[4]=0; print join("\t", @a), "\n";' <SGD_all_ORFs_3prime_UTRs.uniq.bed >SGD_all_ORFs_3prime_UTRs.uniq2.bed 
mafFrags sacCer3 multiz7way SGD_all_ORFs_3prime_UTRs.uniq2.bed test.maf
less test.
less test.,
less test.maf 
grep chrII SGD_all_ORFs_3prime_UTRs.uniq2.bed | head -1
mafFrag sacCer3 multiz7way chrI 100 200 + test2.maf
cat test2.maf 
mafFrag sacCer3 multiz7way chrII 100 200 + test2.maf
cat test2.maf 
mafFrags sacCer3 multiz7way SGD_all_ORFs_3prime_UTRs.uniq2.bed test.maf &
less test.maf 
grep YBL107C test.maf 
grep -A4 YBL107C test.maf 
grep -A5 YBL107C test.maf 
grep -A6 YBL107C test.maf 
maf2clustal.pl <test.maf | less
less test.maf 
maf2clustal.pl <test.maf | less
maf2clustal.pl test.maf | less
less test.maf 
wc -l test.maf 
tail -1 SGD_all_ORFs_3prime_UTRs.uniq2.bed
tail -10 test.maf 
less SGD_all_ORFs_3prime_UTRs.uniq2.bed
les SGD_all_ORFs_3prime_UTRs.uniq2.bed
less test.maf 
mafFrags
less test.maf 
perl -ne '@a=split; $a[4]=0; $a[1]-=1; $a[2]-=1; print join("\t", @a), "\n";' <SGD_all_ORFs_3prime_UTRs.uniq.bed >SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
rm SGD_all_ORFs_3prime_UTRs.uniq2.bed 
mafFetch 
less test2.maf 
rm test2.maf 
less test.maf 
mafFrags sacCer3 multiz7way SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf &
maf2clustal.pl test.maf | less
maf2clustal.pl test.maf | wc -l
maf2clustal.pl SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf | wc -l
maf2clustal.pl SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf | less
nano ~/software/perl/maf2clustal.pl 
maf2clustal.pl SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf | less
nano ~/software/perl/maf2clustal.pl 
maf2clustal.pl SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf | less
nano ~/software/perl/maf2clustal.pl 
maf2clustal.pl SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf | less
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
les SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
mkdir ~/databases
mkdir yeast
mkdir multiz7way
ftp hgdownload.cse.ucsc.edu 
rm upstream*
rm md5sum.txt 
less README.txt 
less 7way.nh 
mafsInRegion 
mafsInRegion SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf ~/databases/yeast/multiz7way/*maf.gz
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
mafsInRegion SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf ~/databases/yeast/multiz7way/*maf.gz
wc -l SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
mafsSpli
mafFetch 
mafFilter 
mafGene 
mafMeFirst 
mafOrder
mafRanges
mafRangesmafSplit
mafSplit
mafSplit SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf ~/databases/yeast/multiz7way/*maf.gz
wc -l SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
tail -2 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
tail -2 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
tail -3 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
tail -6 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
grep 158966 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
les SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
grep YML058W SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed test.bed
grep YML058W SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed >test.bed
cat test.bed 
mafSplit test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
less test.maf 
less test.bed 
less test.maf 
rm test.maf 
mafSplit test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
nano test.bed 
mafSplit test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
nano test.bed 
mafSplit test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
less test.maf
mafsInRegion test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
less test.maf 
grep YML058W SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed >test.bed
mafsInRegion test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
less test.maf 
cat test.maf 
mafsInRegion SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf ~/databases/yeast/multiz7way/*maf.gz
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
less SGD_all_ORFs_3prime_UTRs.uniq.zerobasedchrXVI.99.maf 
mafSplit
less SGD_all_ORFs_3prime_UTRs.uniq.zerobasedchrXVI.99.maf 
grep 948066 SGD_all_ORFs_3prime_UTRs.uniq.bed
grep 948066 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
grep 948065 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
grep 948067 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
less SGD_all_ORFs_3prime_UTRs.uniq.zerobasedchrXVI.99.maf 
grep 384408 SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed 
rm SGD_all_ORFs_3prime_UTRs.uniq.zerobasedchr*
rm testchr*
less test.bed 
mafSplit -byTarget test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
lh *maf
less test000.maf 
rm test0*
mafSplit test.bed test.maf ~/databases/yeast/multiz7way/*maf.gz
less testchrI.00.maf
nano test.bed 
less testchrI.00.maf
lh testchr*
rm testchr*
split 
man split
split SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
man split
split SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
wc -l SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
less xaa
rm xaa xab xac
man split
split -d SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed
less x00
wc -l x*
mafsInRegion x00 x00.maf ~/databases/yeast/multiz7way/*maf.gz
less x00.maf
mafsInRegion
mafsInRegion -keepInitialGaps x00 x00.maf ~/databases/yeast/multiz7way/*maf.gz
less x00.maf
mafFrags
mafsInRegion
mafsInRegion -outDir x00 x00 ~/databases/yeast/multiz7way/*maf.gz
mkdir x00
mafsInRegion -outDir x00 x000 ~/databases/yeast/multiz7way/*maf.gz
less x000/YYIR037W.maf
less x000/YIR037W.maf
less x000/YBL064C.maf
maf2clustal.pl x000/YBL064C.maf
mafFilter 
mafsInRegion -outDir SGD_all_ORFs_3prime_UTRs.uniq.zerobased.bed x001 ~/databases/yeast/multiz7way/*maf.gz
tar xvvzf jydu-maffilter-v1.3.1-9-gc975f4d.tar.gz 
rm jydu-maffilter-v1.3.1-9-gc975f4d.tar.gz 
less INSTALL 
mafFilter 
man mafFilter
less INSTALL 
cmake -DCMAKE_INSTALL_PREFIX=/usr/local
less INSTALL 
sudo apt-get install libbpp-core
uname -a
wget https://launchpad.net/ubuntu/+source/libbpp-core/2.4.0-3ubuntu0.1
rm 2.4.0-3ubuntu0.1 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-core-dev_2.4.0-3ubuntu0.1_amd64.deb
dpkg -i libbpp-core-dev_2.4.0-3ubuntu0.1_amd64.deb 
sudo dpkg -i libbpp-core-dev_2.4.0-3ubuntu0.1_amd64.deb 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-core4_2.4.0-3ubuntu0.1_amd64.deb
sudo dpkg -i libbpp-core4_2.4.0-3ubuntu0.1_amd64.deb 
rm libbpp-core*
cmake -DCMAKE_INSTALL_PREFIX=/usr/local
less INSTALL 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-seq12_2.4.0-2ubuntu0.1_amd64.deb
sudo dpkg -i libbpp-seq12_2.4.0-2ubuntu0.1_amd64.deb 
rm libbpp-seq12_2.4.0-2ubuntu0.1_amd64.deb 
less INSTALL 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-phyl12_2.4.0-1_amd64.deb
sudo dpkg -i libbpp-phyl12_2.4.0-1_amd64.deb 
rm libbpp-phyl12_2.4.0-1_amd64.deb 
less INSTALL 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-phyl-omics3_2.4.0-2_amd64.deb
sudo dpkg -i libbpp-phyl-omics3_2.4.0-2_amd64.deb 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-seq-omics3_2.4.0-2ubuntu0.1_amd64.deb
sudo dpkg -i libbpp-seq-omics3_2.4.0-2ubuntu0.1_amd64.deb 
sudo dpkg -i libbpp-phyl-omics3_2.4.0-2_amd64.deb 
rm libbpp-*
cmake -DCMAKE_INSTALL_PREFIX=/usr/local
less CMakeLists.txt 
les CMakeLists.txt 
wget https://launchpad.net/ubuntu/+archive/primary/+files/libbpp-phyl-omics3_2.4.0-2_amd64.deb
sudo dpkg -i libbpp-phyl-omics3_2.4.0-2_amd64.deb 
rm libbpp-phyl-omics3_2.4.0-2_amd64.deb 
cmake -DCMAKE_INSTALL_PREFIX=/usr/local
cmake -DCMAKE_INSTALL_PREFIX=.
less SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
rm -r jydu-maffilter-c975f4d/
mafFrags sacCer3 multiz7way x00 x00.maf
mafFrags sacCer3 multiz7way x01 x01.maf &
mafFrags sacCer3 multiz7way x02 x02.maf &
less x00.maf
mafFrag
rm test.bed 
rm test.maf 
rm SGD_all_ORFs_3prime_UTRs.uniq.zerobased.maf
grep 169115 x01
grep 169115 SGD_all_ORFs_3prime_UTRs.uniq.bed
grep 169116 SGD_all_ORFs_3prime_UTRs.uniq.bed
nano x01 
mafFrags sacCer3 multiz7way x01 x01.maf &
mkdir split
mv x0* split/
rm -r x000 x001/
maf2clustal.pl x00.maf | less
nano ~/software/perl/maf2clustal.pl 
maf2clustal.pl x00.maf | less
nano ~/software/perl/maf2clustal.pl 
maf2clustal.pl x00.maf | less
nano ~/software/perl/maf2clustal.pl 
maf2clustal.pl x00.maf | less
maf2clustal.pl x00.maf | perl -ne '$seq=<>; print "$_$seq" if $seq !~ /\./;' | less
maf2clustal.pl x00.maf | perl -ne '$seq=<>; print "$_$seq" if $seq !~ /\./;' | wc -l
maf2clustal.pl x00.maf | perl -ne '$seq=<>; print "$_$seq"' | wc -l
maf2clustal.pl x00.maf | perl -ne '$seq=<>; print "$_$seq" if $seq !~ /\./;' | wc -l
maf2clustal.pl x00.maf | perl -ne '$seq=<>; $seq =~ s/\-//d; print "$_$seq" if $seq !~ /\./;' | less
maf2clustal.pl x00.maf | perl -ne '$seq=<>; $seq =~ s/-//d; print "$_$seq" if $seq !~ /\./;' | less
maf2clustal.pl x00.maf | perl -ne '$seq=<>; $seq =~ s/-//g; print "$_$seq" if $seq !~ /\./;' | less
maf2clustal.pl x00.maf | perl -ne '$seq=<>; $seq =~ s/-//g; print "$_$seq" if $seq !~ /\./;' | wc -l
maf2clustal.pl x00.maf | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' | wc -l
maf2clustal.pl x00.maf | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' >all3utrs.txt
less all3utrs.txt 
maf2clustal.pl x01.maf | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' >>all3utrs.txt
maf2clustal.pl x02.maf | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' >>all3utrs.txt
wc -l all3utrs.txt 
less x00
less x00.maf 
grep score x00.maf | wc -l
grep score x01.maf | wc -l
grep score x02.maf | wc -l
calc 10428/2849
less all3utrs.txt 
perl -ne 'print $_ if length($_) < 10;' <all3utrs.txt | less
perl -ne 'print $_ if length($_) < 20;' <all3utrs.txt | less
perl -ne 'print $_ if length($_) >= 20;' <all3utrs.txt >all3utrs.gt20.txt 
less all3utrs.gt20.txt 
ls *bed
less yeastorfs.bed 
wc -l yeastorfs.bed 
hgrep yeastorfs.bed 
mkdir split
split -d ../yeastorfs.bed 
wc -l *
for x in x*; do { echo "mafFrags sacCer3 multiz7way $x $x.maf"; } done
for x in x*; do { mafFrags sacCer3 multiz7way $x $x.maf & } done
for x in x*; do { mafFrags sacCer3 multiz7way $x $x.maf &; } done
for x in x*; do { mafFrags sacCer3 multiz7way $x {$x}.maf & } done
rm *maf
for x in x*; do { mafFrags sacCer3 multiz7way $x $x\.maf & } done
rm *maf
for x in x*; do mafFrags sacCer3 multiz7way $x $x.maf & done
rm *maf
for x in x*; do echo mafFrags sacCer3 multiz7way $x $x.maf & done
for x in x*; do echo mafFrags sacCer3 multiz7way $x $x.maf; done
for x in x*; do echo "mafFrags sacCer3 multiz7way $x $x.maf &"; done
mafFrags sacCer3 multiz7way x00 x00.maf &
mafFrags sacCer3 multiz7way x01 x01.maf &
mafFrags sacCer3 multiz7way x02 x02.maf &
mafFrags sacCer3 multiz7way x03 x03.maf &
mafFrags sacCer3 multiz7way x04 x04.maf &
mafFrags sacCer3 multiz7way x05 x05.maf &
mafFrags sacCer3 multiz7way x06 x06.maf &
mafFrags sacCer3 multiz7way x02 x02.maf
hgrep zerobased
perl -ne '@a=split; $a[4]=0; $a[1]-=1; $a[2]-=1; print join("\t", @a), "\n";' <yeastorfs.bed >yeastorfs.zerobased.bed 
rm x*
split -d ../yeastorfs.zerobased.bedbed 
split -d ../yeastorfs.zerobased.bed
for x in x*; do mafFrags sacCer3 multiz7way $x $x.maf & done
for x in x*.maf; do { maf2clustal.pl $x | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' >>allorfs.txt; } done
less allorfs.txt 
less ../yeastorfs.bed 
mafFrag sacCer3 multiz7way chrI    24000   27968  + test2.maf
less test2.maf 
grep 24000 ../yeastorfs.bed 
mafFrag sacCer3 multiz7way chrI    24000   27968  - test2.maf
less test2.maf 
grep -a2 24000 ../yeastorfs.bed 
mafFrag sacCer3 multiz7way chrI 21566 21850 + test2.maf
cat test2.maf 
mafFrag sacCer3 multiz7way chrI 21566 21850 + test2.maf
cat test2.maf 
perl -ne '@a=split; $a[4]=0; if ($str eq "+"){ $a[1]-=1;} else { $a[2]-=1; } print join("\t", @a), "\n";' <yeastorfs.bed >yeastorfs.zerobased.bed 
head -3 yeastorfs.zerobased.bed
head -1- yeastorfs.zerobased.bed
head -10 yeastorfs.zerobased.bed
mafFrag sacCer3 multiz7way 130799 131982 + test2.maf
mafFrag sacCer3 multiz7way chrI 130799 131982 + test2.maf
cat test2.maf 
perl -ne '@a=split; $a[4]=0; if ($str eq "+"){ $a[2]-=1;} else { $a[1]-=1; } print join("\t", @a), "\n";' <yeastorfs.bed >yeastorfs.zerobased.bed 
head -10 yeastorfs.zerobased.bed
mafFrag sacCer3 multiz7way chrI 130798 131983 + test2.maf
cat test2.maf 
head -10 yeastorfs.zerobased.bed
mafFrag sacCer3 multiz7way chrI 334 649 + test2.maf
cat test2.maf 
mafFrag sacCer3 multiz7way chrI 1806 2169 - test2.maf
cat test2.maf 
split -d ../yeastorfs.zerobased.bed
rm *maf
for x in x*; do mafFrags sacCer3 multiz7way $x $x.maf & done
less x00.maf 
~/software/StarORF.bin
chmod +x ~/software/StarORF.bin
~/software/StarORF.bin
less ~/software/StarORF.bin 
sudo apt-get install java
sudo apt-get install java-common
java 
less ~/software/StarORF.bin 
~/software/StarORF.bin 
les ~/software/StarORF.bin 
sudo apt-get install locate
~/software/StarORF.bin 
sudo apt install default-jre
java -version
sudo add-apt-repository ppa:webupd8team/java
sudo apt update
sudo apt install oracle-java8-installer
apt list --upgradable
sudo apt install oracle-java8-installer
sudo apt-get install oracle-java8-installer
sudo apt-get install oracle-java8-installersudo apt-get install oracle-java8-set-default
sudo apt-get install oracle-java8-set-defaultsu -
echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee /etc/apt/sources.list.d/webupd8team-java.list
echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys EEA14886
su -
cat >orffinder.py
python orffinder.py 
which python
python orffinder.py 
python
/usr/bin/python2 orffinder.py 
less orffinder.py 
/usr/bin/python2 orffinder.py 
wget ftp://emboss.open-bio.org/pub/EMBOSS/EMBOSS-6.5.7.tar.gz
less em
wget ftp://emboss.open-bio.org/pub/EMBOSS/EMBOSS-6.6.0.tar.gz
tar xvvzf EMBOSS-6.6.0.tar.gz 
sudo apt-get install emboss
sudo apt-get install emboss-lib
sudo apt autoremove
embossversion 
emboss
embossupdate 
embossdata
needle
getorf 
rm -r EMBOSS-6.6.0*
less x03
less allorfs.txt 
less x00.maf 
for x in x*.maf; do { maf2clustal.pl $x | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' >$x.seq.txt; } done
for x in x*.maf; do { maf2clustal.pl $x | perl -ne '$seq=<>; $seq =~ s/-//g; print "$seq" if $seq !~ /\./;' >$x.seq.txt & } done
less x00.maf.seq.txt
head -1 x02.maf.seq.txt >test.txt
getorf test.txt 
getorf <test.txt 
nano 
nano test.txt 
getorf <test.txt 
getorf test.txt 
getorf
git clone https://github.com/sestaton/HMMER2GO.git
less INSTALL.md 
sudo apt-get install -y emboss zlib1g-dev libxml2-dev
less INSTALL.md 
curl -L cpanmin.us | perl - git://github.com/sestaton/HMMER2GO.git
less /home/vagar/.cpanm/work/1579118325.26316/build.log
sudo apt-get install hmmer
curl -L cpanmin.us | perl - git://github.com/sestaton/HMMER2GO.git
perl Makefile.PL
make 
make test
make install
sudo make install
hmmer2go 
hmmer2go getorf
rm -r HMMER2GO/
hmmer2go getorf -c -t 2 -l 100 test
hmmer2go getorf -c -t 2 -l 100 test.txt 
hmmer2go getorf -c -t 2 -l 100 <test.txt 
hmmer2go getorf -c -t 2 -l 100 -i test.txt 
hmmer2go getorf -c -t 2 -l 100 -i test.txt -o test.out
cat test.out 
for x in x*.maf; do { maf2clustal.pl $x | perl -ne '$seq=<>; $seq =~ s/-//g; chomp $seq; print "$_$seq" if $seq !~ /\./;' >$x.seq.fa & } done
for x in x*.maf; do { maf2clustal.pl $x | perl -ne '$seq=<>; $seq =~ s/-//g; chomp $seq; print "$_$seq\n" if $seq !~ /\./;' >$x.seq.fa & } done
less x00.maf.seq.fa
hmmer2go getorf -c -t 2 -l 100 -i x00.maf.seq.fa -o test.out
rm test.out 
hmmer2go getorf -c -t 2 -l 100 -i x00.maf.seq.fa -o test.out
less test.ot
less test.out 
JOBS
hmmer2go getorf -c -l 100 -i x00.maf.seq.fa -o test.out2
less test.out2 
qhmmer2go getorf --man
hmmer2go getorf --man
sudo apt-get install perl-doc
hmmer2go getorf --man
hmmer2go getorf -c -t 3 -l 100 -i x00.maf.seq.fa -o test.out3
less test.out3 
JOBS
hmmer2go getorf --man
hmmer2go getorf -c -t 3 -l 100 -i x00.maf.seq.fa -s -o test.out4
less test.out4
hmmer2go getorf --man
less test.out4
hmmer2go getorf -c -t 3 -l 100 -i x00.maf.seq.fa -o test.out5
less test.out5 
less x00.maf.seq.fa 
less test.out5 
hmmer2go getorf --man
less test.out5 
less x00.maf.seq.fa 
grep -C 5 387 x00.maf.seq.fa | less
grep -C 2 387 x00.maf.seq.fa | less
grep -B 1 387 x00.maf.seq.fa | less
grep -B 1 387 x00.maf.seq.fa | grep ATGATG | less
less x00.maf.seq.fa 
rm test.*
rm *.txt
rm orffinder.py 
less x00.maf.seq_54_1cBE.fa 
less x00.maf.seq_*
rm x00.maf.seq_*
for x in x*.maf; do { maf2clustal.pl $x | perl -ne '$seq=<>; $seq =~ s/-//g; chomp $seq; print "$_$seq\n" if $seq !~ /\./;' >$x.fa & } done
rm *seq.fa
for x in x*.maf.fa; do { hmmer2go getorf -c -t 3 -l 100 -i $x -o $x.codons.fa && unjustify_fasta.pl $x.codons.fa & } done
hmmer2go getorf --man
less x02.maf.fa.codons.fa
less x06.maf.fa.codons.fa.prepared
LS
JOBS
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($start, $stop, $start1, $stop1) =~ /()()()()/;; print "";' <x06.maf.fa.codons.fa.prepared | less
head -2 x06.maf.fa.codons.fa.prepared
head -4 x06.maf.fa.codons.fa.prepared
less x06.maf.fa.codons.fa.prepared
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) =~ /_1-(\d+)_\d+ \[(\d+) - (\d+)\]/;; print "$len, $start1, $stop1";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/;; print "$len, $start1, $stop1";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print "$len, $start1, $stop1";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print "$len, $start1, $stop1\n";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print "$len, $start1, $stop1\n".(($stop1-$start1)/$len)."\n";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print (($stop1-$start1)/$len)."\n";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print (($stop1-$start1)/$len)."\n\n";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print (($stop1-$start1)/$len)."\n\r";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print "\n".(($stop1-$start1)/$len)."\n";' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print "\n".(($stop1-$start1)/$len);' <x06.maf.fa.codons.fa.prepared | less
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); print "\n".(($stop1-$start1)/$len);' <x06.maf.fa.codons.fa.prepared >fractions.txt
R
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); $seq =~ s/\s+$//; print "$_$seq\n" if (($stop1-$start1)/$len) >= 0.9;' <x06.maf.fa.codons.fa.prepared | less
less ../yeastorfs.codons.txt 
hgrep yeastorfs.codons.encoded.txt
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); $seq =~ s/\s+$//; print "$seq\n" if (($stop1-$start1)/$len) >= 0.9;' <x06.maf.fa.codons.fa.prepared | perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n\n";' <yeastorfs.txt >yeastorfs.codons.encoded.txt 
perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); $seq =~ s/\s+$//; print "$seq\n" if (($stop1-$start1)/$len) >= 0.9;' <x06.maf.fa.codons.fa.prepared | perl -ne 'chomp; for($i=0; $i < length($_)-3; $i+=3){ print substr($_,$i,3)." "; } print substr($_,$i,3)."\n\n";' >yeastorfs.codons.encoded.txt 
less yeastorfs.codons.encoded.txt 
for x in x*.maf.fa.prepared; do { perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); $seq =~ s/\s+$//; print "$seq\n" if (($stop1-$start1)/$len) >= 0.9;' <$x >$x.correctLen.txt & } done
for x in x*.maf.fa.codons.fa.prepared; do { perl -ne '$seq = <>; next if $_ =~ /REVERSE/; ($len, $start1, $stop1) = ($_ =~ /\_1-(\d+)\_\d+ \[(\d+) - (\d+)\]/); $seq =~ s/\s+$//; print "$seq\n" if (($stop1-$start1)/$len) >= 0.9;' <$x >$x.correctLen.txt & } done
less x00.maf.fa.codons.fa.prepared.correctLen.txt
LS
cat *correctLen* >ALLFINAL.txt
less ALLFINAL.txt 
wc -l ALLFINAL.txt 
wc -l ../yeastorfs.codons.txt
calc 20948/6692
less ~/.bashrc 
ls ~/predict_expression/
less ~/.bashrc 
gitadd 
ls google-drive/
rmdir google-drive/
lh ALBERT/yeastTest
less ALBERT/run_regression.py 
less ALBERT/classifier_utils.py 
mkdir unsupervised_learning
mv unsupervised_learning/ semisupervised_learning/
ln -s ../sentencepiece/yeastTest/ .
ln -s ../ALBERT/yeast* .
ln -s ../ALBERT/*sb .
ln -s ../ALBERT/*json .
ln -s ../ALBERT/classifier_utils.py 
ln -s ../ALBERT/run_regression.py 
ln -s ../ALBERT/run.sh 
less ~/.bashrc 
grep gitadd ~/.bashrc 
find /home/vagar/semisupervised_learning/ -name '*.py' -or -name '*.ipynb' -or -name '*.sb' -or -name '*.gin' -or -name '*.joblib' -or -name '*.sh'
nano test.txt 
nano ~/.bashrc 
find /home/vagar/semisupervised_learning/ -name '*.py' -or -name '*.ipynb' -or -name '*.sb' -or -name '*.gin' -or -name '*.joblib' -or -name '*.sh' -or -name '*.json'
mkdir google3
mkdir configs
touch configs/sweep.yaml
touch configs/transformer_expression_prediction_8gb.gin
touch BUILD
touch train_on_2x2.sh
touch vikramtrax.py
nano ~/.bashrc 
