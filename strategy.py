#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_Pullback_TrendFilter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA trend on 1d
    price_series = pd.Series(close)
    change = abs(price_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.finfo(float).eps)
    er = er.fillna(0)
    sc = (er * 0.06 + 0.06) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(close > kama, 1, -1)
    
    # RSI(14) for pullback
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    weekly_ma = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean()
    weekly_ma_values = weekly_ma.values
    weekly_ma_aligned = align_htf_to_ltf(prices, df_1w, weekly_ma_values)
    
    # Volume filter (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or 
            np.isnan(weekly_ma_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, weekly trend up, RSI pullback <40, volume surge
            if (kama_dir[i] == 1 and 
                close[i] > weekly_ma_aligned[i] and 
                rsi[i] < 40 and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, weekly trend down, RSI pullback >60, volume surge
            elif (kama_dir[i] == -1 and 
                  close[i] < weekly_ma_aligned[i] and 
                  rsi[i] > 60 and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI reverts to middle or trend changes
            if position == 1:
                if rsi[i] > 60 or kama_dir[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if rsi[i] < 40 or kama_dir[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals