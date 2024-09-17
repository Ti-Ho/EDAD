python -u run_EDAD.py --dataset=NON10 >> log_EDAD/NON10_log.txt
python -u run_EDAD.py --dataset=NON10_5NOISE >> log_EDAD/NON10_5_log.txt
python -u run_EDAD.py --dataset=NON10_10NOISE >> log_EDAD/NON10_10_log.txt
python -u run_EDAD.py --dataset=NON10_15NOISE >> log_EDAD/NON10_15_log.txt
python -u run_EDAD.py --dataset=NON10_20NOISE >> log_EDAD/NON10_20_log.txt
python -u run_EDAD.py --dataset=NON10_30NOISE >> log_EDAD/NON10_30_log.txt

python -u run_EDAD.py --dataset=NON1388 >> log_EDAD/NON1388_log.txt
python -u run_EDAD.py --dataset=NON1388_5NOISE >> log_EDAD/NON1388_5_log.txt
python -u run_EDAD.py --dataset=NON1388_10NOISE >> log_EDAD/NON1388_10_log.txt
python -u run_EDAD.py --dataset=NON1388_15NOISE >> log_EDAD/NON1388_15_log.txt
python -u run_EDAD.py --dataset=NON1388_20NOISE >> log_EDAD/NON1388_20_log.txt
python -u run_EDAD.py --dataset=NON1388_30NOISE >> log_EDAD/NON1388_30_log.txt

python -u run_EDAD.py --dataset=SP500N --enc_in=483 --dec_in=483 --c_out=483 >> log_EDAD/SP500N_log.txt
python -u run_EDAD.py --dataset=SP500N_5NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log_EDAD/SP500N_5_log.txt
python -u run_EDAD.py --dataset=SP500N_10NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log_EDAD/SP500N_10_log.txt
python -u run_EDAD.py --dataset=SP500N_15NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log_EDAD/SP500N_15_log.txt
python -u run_EDAD.py --dataset=SP500N_20NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log_EDAD/SP500N_20_log.txt
python -u run_EDAD.py --dataset=SP500N_30NOISE --enc_in=483 --dec_in=483 --c_out=483 >> log_EDAD/SP500N_30_log.txt
