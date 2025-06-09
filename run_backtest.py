import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import optuna
import json
import time

# Strategy class: трендовый прорыв + ATR стоп/профит + фиксированный риск
class EMARSI(bt.Strategy):
    params = dict(
        ema_fast=50,
        ema_mid=100,
        ema_slow=200,
        rsi_period=14,
        rsi_low=30,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        atr_period=14,
        risk_per_trade=0.01,
    )

    def __init__(self):
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_mid  = bt.ind.EMA(self.data.close, period=self.p.ema_mid)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.rsi      = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.atr      = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.entry_price = self.stop_price = self.take_price = None

    def next(self):
        # Выход по SL/TP
        if self.position.size:
            price = self.data.close[0]
            if price <= self.stop_price or price >= self.take_price:
                self.close()
            return

        # Вход: тренд + RSI перепроданности
        if self.ema_fast[0] > self.ema_mid[0] > self.ema_slow[0] and self.rsi[0] < self.p.rsi_low:
            cash = self.broker.get_cash()
            risk_amt = cash * self.p.risk_per_trade
            stop_dist = self.atr[0] * self.p.sl_atr_mult
            if stop_dist <= 0:
                return
            size = int(risk_amt / stop_dist)
            if size <= 0:
                return

            entry = self.data.close[0]
            self.entry_price = entry
            self.stop_price  = entry - stop_dist
            self.take_price  = entry + self.atr[0] * self.p.tp_atr_mult
            self.buy(size=size)

# Функция single walk-forward для одного окна

def walk_forward(params, df, train_start, train_end, test_start, test_end):
    # train
    train_feed = bt.feeds.PandasData(
        dataname=df,
        fromdate=train_start,
        todate=train_end,
        timeframe=bt.TimeFrame.Minutes,
        compression=5
    )
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EMARSI, **params)
    cerebro.adddata(train_feed)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.00002)
    cerebro.run()

    # test
    test_feed = bt.feeds.PandasData(
        dataname=df,
        fromdate=test_start,
        todate=test_end,
        timeframe=bt.TimeFrame.Minutes,
        compression=5
    )
    cerebro_test = bt.Cerebro()
    cerebro_test.addstrategy(EMARSI, **params)
    cerebro_test.adddata(test_feed)
    cerebro_test.broker.setcash(100_000)
    cerebro_test.broker.setcommission(commission=0.00002)
    cerebro_test.addanalyzer(bt.analyzers.Returns, _name='returns')
    res = cerebro_test.run()[0]
    analysis = res.analyzers.returns.get_analysis()
    return analysis.get('rnorm', analysis.get('rtot', 0))

# Multi walk-forward: скользящие окна

def multi_walk_forward(params, df, train_years=5, test_years=1, step_months=12):
    results = []
    start = df.index.min()
    end   = df.index.max()

    current_test_start = start + pd.DateOffset(years=train_years)
    total_windows = 0
    # посчитаем число окон
    temp = current_test_start
    while temp + pd.DateOffset(years=test_years) <= end:
        total_windows += 1
        temp += pd.DateOffset(months=step_months)

    processed = 0
    while current_test_start + pd.DateOffset(years=test_years) <= end:
        train_start = current_test_start - pd.DateOffset(years=train_years)
        train_end   = current_test_start - pd.Timedelta(days=1)
        test_start  = current_test_start
        test_end    = current_test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)

        ann = walk_forward(params, df, train_start, train_end, test_start, test_end)
        processed += 1
        elapsed = time.time() - multi_walk_forward.start_time
        est_total = elapsed / processed * total_windows
        rem = est_total - elapsed
        print(f'Window {processed}/{total_windows}, elapsed {elapsed:.1f}s, est remain {rem:.1f}s')

        results.append({'train': (train_start.date(), train_end.date()),
                        'test':  (test_start.date(), test_end.date()),
                        'ann_ret': ann})

        current_test_start += pd.DateOffset(months=step_months)

    return pd.DataFrame(results)

# Optuna objective: среднее по окнам

def objective(trial, df):
    params = {
        'ema_fast': trial.suggest_int('ema_fast', 20, 100, step=10),
        'ema_mid':  trial.suggest_int('ema_mid', 80, 200, step=20),
        'ema_slow': trial.suggest_int('ema_slow', 150, 300, step=50),
        'rsi_low': trial.suggest_int('rsi_low', 20, 40, step=5),
        'sl_atr_mult': trial.suggest_float('sl_atr_mult', 1.0, 2.0, step=0.1),
        'tp_atr_mult': trial.suggest_float('tp_atr_mult', 2.0, 4.0, step=0.5),
    }
    multi_walk_forward.start_time = time.time()
    df_res = multi_walk_forward(params, df)
    mean_ann = df_res['ann_ret'].mean()
    return -mean_ann

# Основной запуск вручную

def main():
    df = pd.read_csv('EURUSD_iM5.csv', header=None, encoding='utf-8')
    df.columns = ['datetime','open','high','low','close','volume','dummy']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y.%m.%d %H:%M')
    df.set_index('datetime', inplace=True)

    study = optuna.create_study(direction='minimize')
    start = time.time()
    def progress(study, trial):
        elapsed = time.time() - start
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        total = 50
        est_total = elapsed / completed * total if completed else 0
        rem = est_total - elapsed
        print(f'Trial {completed}/{total}, elapsed {elapsed:.1f}s, est remain {rem:.1f}s')

    study.optimize(lambda t: objective(t, df), n_trials=50, callbacks=[progress])

    best = study.best_params
    best_ret = -study.best_value
    print(f"Лучшие параметры: {best}, средний Annual Return={best_ret*100:.2f}%")
    with open('best_params.json', 'w') as f:
        json.dump({'params': best, 'mean_ret': best_ret}, f)

    df_windows = multi_walk_forward(best, df)
    print(df_windows)

if __name__ == '__main__':
    main()
