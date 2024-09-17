python -u run.py --dataset=NON10 >> log/NON10_log.txt
python -u run.py --dataset=NON10_5NOISE >> log/NON10_5_log.txt
python -u run.py --dataset=NON10_10NOISE >> log/NON10_10_log.txt
python -u run.py --dataset=NON10_15NOISE >> log/NON10_15_log.txt
python -u run.py --dataset=NON10_20NOISE >> log/NON10_20_log.txt
python -u run.py --dataset=NON10_30NOISE >> log/NON10_30_log.txt

python -u run.py --dataset=NON1388 >> log/NON1388_log.txt
python -u run.py --dataset=NON1388_5NOISE >> log/NON1388_5_log.txt
python -u run.py --dataset=NON1388_10NOISE >> log/NON1388_10_log.txt
python -u run.py --dataset=NON1388_15NOISE >> log/NON1388_15_log.txt
python -u run.py --dataset=NON1388_20NOISE >> log/NON1388_20_log.txt
python -u run.py --dataset=NON1388_30NOISE >> log/NON1388_30_log.txt

python -u run.py --dataset=SP500N --enc_in=483 --dec_in=483 --c_out=483 >> log/SP500N_log.txt
python -u run.py --dataset=SP500N_5NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log/SP500N_5_log.txt
python -u run.py --dataset=SP500N_10NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log/SP500N_10_log.txt
python -u run.py --dataset=SP500N_15NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log/SP500N_15_log.txt
python -u run.py --dataset=SP500N_20NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log/SP500N_20_log.txt
python -u run.py --dataset=SP500N_30NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log/SP500N_30_log.txt
