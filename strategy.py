#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_12hVolume_Spike_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h volume for spike detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma10_12h = pd.Series(vol_12h).rolling(window=10, min_periods=10).mean().values
    vol_spike_12h = vol_12h > (2.0 * vol_ma10_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TRIX on 4h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    trix_sma9 = pd.Series(trix).rolling(window=9, min_periods=9).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_sma9[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, daily uptrend, volume spike
            long_cond = (trix_sma9[i] > 0 and trix_sma9[i-1] <= 0 and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        vol_spike_12h_aligned[i])
            
            # Short: TRIX crosses below zero, daily downtrend, volume spike
            short_cond = (trix_sma9[i] < 0 and trix_sma9[i-1] >= 0 and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         vol_spike_12h_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_sma9[i] < 0 and trix_sma9[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_sma9[i] > 0 and trix_sma9[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals