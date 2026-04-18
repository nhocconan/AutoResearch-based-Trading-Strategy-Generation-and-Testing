#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_1dTrendFilter_v1
Hypothesis: TRIX (triple EMA) momentum with volume spike and 1d EMA trend filter captures trend continuation in both bull and bear markets. Designed for 15-25 trades/year on 12h timeframe to minimize fee drag while capturing major moves.
"""

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
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TRIX: Triple EMA (15,15,15) - momentum oscillator
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_values = trix.fillna(0).values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(trix_values[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_values[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TRIX positive (bullish momentum) + above 1d EMA + volume spike
            if trix_val > 0.1 and close[i] > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative (bearish momentum) + below 1d EMA + volume spike
            elif trix_val < -0.1 and close[i] < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX turns negative OR price breaks below 1d EMA
            if trix_val < -0.05 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX turns positive OR price breaks above 1d EMA
            if trix_val > 0.05 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_TRIX_VolumeSpike_1dTrendFilter_v1"
timeframe = "12h"
leverage = 1.0