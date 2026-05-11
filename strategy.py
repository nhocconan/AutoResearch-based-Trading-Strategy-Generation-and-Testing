#!/usr/bin/env python3
name = "1d_TRIX_Volume_Spike_Trend_1w"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w trend: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # TRIX: Triple EMA of ROC
    roc = np.zeros(n)
    for i in range(1, n):
        if close[i-1] != 0:
            roc[i] = (close[i] - close[i-1]) / close[i-1]
    
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.where(ema2 != 0, (ema3 - ema2) / ema2 * 100, 0)
    
    # Volume spike: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(45, 20)  # ROC(1) + 3*EMA15 + vol MA20
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX turning up + volume spike + price above 1w EMA34
            if trix[i] > trix[i-1] and vol_spike[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down + volume spike + price below 1w EMA34
            elif trix[i] < trix[i-1] and vol_spike[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX turns down
            if trix[i] < trix[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX turns up
            if trix[i] > trix[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals