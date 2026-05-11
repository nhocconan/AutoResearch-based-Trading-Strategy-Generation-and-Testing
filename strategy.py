#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # TRIX (15-period) - TRIPLE EMA of 1-period ROC
    close_series = pd.Series(close)
    roc = close_series.pct_change(1)  # 1-period rate of change
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3 * 100  # scale for readability
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    
    # 12h EMA21 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for TRIX calculation)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line AND above 12h EMA21 (uptrend) AND volume spike
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line AND below 12h EMA21 (downtrend) AND volume spike
            elif trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line OR below 12h EMA21 (trend change)
            if trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: TRIX crosses above signal line OR above 12h EMA21 (trend change)
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals