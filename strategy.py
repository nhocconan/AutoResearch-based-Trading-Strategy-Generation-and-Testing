#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX(12) calculation
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # percent change
    trix_signal = trix.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_hist = trix - trix_signal
    trix_values = trix_hist.values
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: volume > 1.5 * 20-period SMA of volume
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_sma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(36, 20)  # TRIX needs 36 for stability
    
    for i in range(start_idx, n):
        if np.isnan(trix_values[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX histogram crosses above zero + price above 1d EMA34 + volume spike
            if trix_values[i] > 0 and trix_values[i-1] <= 0 and close[i] > ema_34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX histogram crosses below zero + price below 1d EMA34 + volume spike
            elif trix_values[i] < 0 and trix_values[i-1] >= 0 and close[i] < ema_34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX histogram crosses below zero
            if trix_values[i] < 0 and trix_values[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX histogram crosses above zero
            if trix_values[i] > 0 and trix_values[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals