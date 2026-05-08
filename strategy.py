#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_KAMA_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d trend: EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume ratio
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 6h KAMA (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 6h volume ratio
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume / vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(vol_ratio_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA + daily trend up + volume spike
            if (close[i] > kama[i] and
                close[i] > ema_34_aligned[i] and
                vol_ratio_6h[i] > 1.5 and
                vol_ratio_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA + daily trend down + volume spike
            elif (close[i] < kama[i] and
                  close[i] < ema_34_aligned[i] and
                  vol_ratio_6h[i] > 1.5 and
                  vol_ratio_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < KAMA or daily trend down
            if (close[i] < kama[i] or
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA or daily trend up
            if (close[i] > kama[i] or
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals