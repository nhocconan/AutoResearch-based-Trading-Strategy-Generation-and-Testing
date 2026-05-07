#!/usr/bin/env python3
# 4H_TRIX_Signal_With_Volume_Confirmation
# Hypothesis: Uses TRIX(12) momentum on 4h timeframe with volume spike confirmation
# and 12h EMA50 trend filter. TRIX captures momentum shifts while volume and trend
# filters reduce false signals. Works in both bull and bear markets by taking long
# when TRIX crosses above zero in uptrend and short when TRIX crosses below zero
# in downtrend. Target: 20-50 trades per year with size 0.25 to avoid fee drag.

name = "4H_TRIX_Signal_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close: TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    # Using 12-period EMA triple smoothed
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # Percentage change of triple EMA
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 36  # Need 12*3 = 36 periods for triple EMA + 20 for vol MA
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get TRIX value (handle both Series and array)
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        trix_prev = trix.iloc[i-1] if hasattr(trix, 'iloc') else trix[i-1]
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + Uptrend (close > EMA50_12h) + volume spike
            if (trix_prev <= 0 and trix_val > 0 and 
                close[i] > ema50_12h_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + Downtrend (close < EMA50_12h) + volume spike
            elif (trix_prev >= 0 and trix_val < 0 and 
                  close[i] < ema50_12h_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend turns down
            if trix_val < 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend turns up
            if trix_val > 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals