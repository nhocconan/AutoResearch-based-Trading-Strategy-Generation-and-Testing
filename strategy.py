#!/usr/bin/env python3
# 12h_1d_1w_trix_volume_regime
# Hypothesis: Use TRIX momentum on 1d timeframe to detect trend, confirmed by volume surge and 1w trend filter.
# Enter long when TRIX crosses above zero with volume surge and 1w uptrend; short when TRIX crosses below zero with volume surge and 1w downtrend.
# Exit when TRIX crosses back across zero or 1w trend reverses.
# Uses volume filter to avoid false signals and weekly EMA for trend filter.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_trix_volume_regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: triple-smoothed EMA of % change
    # TRIX = EMA(EMA(EMA(close, period), period), period)
    # Then: TRIX_pct = (TRIX - TRIX.shift(1)) / TRIX.shift(1) * 100
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Calculate % change of triple smoothed EMA
    trix = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
        else:
            trix[i] = 0
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(trix_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero or weekly trend breaks (price < weekly EMA21)
            if trix_aligned[i] < 0 or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero or weekly trend breaks (price > weekly EMA21)
            if trix_aligned[i] > 0 or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX crosses above zero with volume surge and weekly uptrend
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and vol_surge and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX crosses below zero with volume surge and weekly downtrend
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and vol_surge and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals