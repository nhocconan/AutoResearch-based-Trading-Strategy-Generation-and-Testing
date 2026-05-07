#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter (on daily data)
    ema_50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly EMA34 for trend filter (on weekly data)
    w_close = df_1w['close'].values
    ema_34_1w = pd.Series(w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_d[i]) or np.isnan(ema_34_1w_d[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: weekly uptrend (price > weekly EMA34) AND daily uptrend (price > daily EMA50) with volume
            if close[i] > ema_34_1w_d[i] and close[i] > ema_50_d[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price < weekly EMA34) AND daily downtrend (price < daily EMA50) with volume
            elif close[i] < ema_34_1w_d[i] and close[i] < ema_50_d[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend reversal or volume divergence
            if close[i] < ema_34_1w_d[i] or close[i] < ema_50_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend reversal or volume divergence
            if close[i] > ema_34_1w_d[i] or close[i] > ema_50_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals