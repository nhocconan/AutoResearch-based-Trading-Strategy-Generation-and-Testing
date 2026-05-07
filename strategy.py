#!/usr/bin/env python3
name = "6h_TRIX_ZeroCross_1dTrend_Volume"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop for TRIX and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX (15-period triple EMA) on daily closes
    close_1d = pd.Series(df_1d['close'])
    ema1 = close_1d.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    trix = trix_raw.values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # Align TRIX and EMA to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume and daily uptrend
            trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if trix_cross_up and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume and daily downtrend
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or volume drops
            if trix_aligned[i] < 0 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or volume drops
            if trix_aligned[i] > 0 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX zero-cross on daily with volume and trend filter
# - TRIX (triple EMA momentum) identifies momentum shifts via zero cross
# - Daily TRIX zero-cross + volume spike confirms institutional participation
# - Daily EMA(34) trend filter ensures trades align with higher timeframe trend
# - Works in both bull (buy zero-cross in uptrend) and bear (sell zero-cross in downtrend)
# - Volume confirmation (1.8x average) reduces false signals
# - Exit on TRIX reversal or volume decline prevents overstaying
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Novel combination: TRIX momentum (1d) + volume (6h) + trend (1d) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits