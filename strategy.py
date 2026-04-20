#!/usr/bin/env python3
# 12h_TRIX_VolumeSpike_Regime
# Hypothesis: TRIX (15-period) captures momentum changes while reducing noise. 
# Combined with volume spikes (>2x 20-period average) and chop regime filter (CHOP > 61.8 = ranging, < 38.2 = trending), 
# this strategy enters long when TRIX crosses above zero in trending markets with volume confirmation, 
# and short when TRIX crosses below zero. Designed for 12h timeframe to limit trades (target: 50-150/4 years) 
# and avoid fee drag. Works in bull/bear markets via regime adaptation.

name = "12h_TRIX_VolumeSpike_Regime"
timeframe = "12h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX: 1-period ROC of triple-smoothed EMA (15-period)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change(1)).values  # ROC of triple EMA
    
    # Chopiness Index (14-period) - regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    sum_tr = atr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 2.0)
    
    # Align TRIX and Chop to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume + trending regime (CHOP < 38.2)
            if i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and volume_filter[i] and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume + trending regime (CHOP < 38.2)
            elif i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and volume_filter[i] and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if TRIX crosses below zero OR chop indicates ranging (CHOP > 61.8)
            if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if TRIX crosses above zero OR chop indicates ranging (CHOP > 61.8)
            if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals