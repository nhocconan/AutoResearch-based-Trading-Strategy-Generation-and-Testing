#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TRIX_1wTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # TRIX on 1w close (12-period EMA triple)
    close_1w_series = pd.Series(close_1w)
    ema1 = close_1w_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    trix_values = trix.fillna(0).values
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 36  # 12*3 for TRIX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TRIX > 0 (bullish momentum), volume spike
            long_cond = (trix_aligned[i] > 0) and volume_spike[i]
            # Short conditions: TRIX < 0 (bearish momentum), volume spike
            short_cond = (trix_aligned[i] < 0) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below 0 (momentum shift)
            if trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above 0 (momentum shift)
            if trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals