#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_ZeroCross_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate TRIX (15-period EMA applied 3 times, then % change)
    # TRIX = 100 * (EMA3(EMA2(EMA1(close, 15), 15), 15) - prev) / prev
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value has no previous
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_raw[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, 12h uptrend, volume spike
            long_cond = (trix_raw[i] > 0 and trix_raw[i-1] <= 0 and
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: TRIX crosses below zero, 12h downtrend, volume spike
            short_cond = (trix_raw[i] < 0 and trix_raw[i-1] >= 0 and
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_raw[i] < 0 and trix_raw[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_raw[i] > 0 and trix_raw[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals