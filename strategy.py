#!/usr/bin/env python3
# 1d_1w_triple_sma_cross_v1
# Hypothesis: Long when price > SMA10 > SMA30 and 1w SMA10 > SMA30, short when price < SMA10 < SMA30 and 1w SMA10 < SMA30.
# Uses 1d SMA10/30 crossover with 1w trend filter to avoid counter-trend trades.
# Target: 10-20 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_triple_sma_cross_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d SMA10 and SMA30
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    sma30 = pd.Series(close).rolling(window=30, min_periods=30).mean().values
    
    # Volume filter: 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    sma30_1w = pd.Series(close_1w).rolling(window=30, min_periods=30).mean().values
    sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma10_1w)
    sma30_1w_aligned = align_htf_to_ltf(prices, df_1w, sma30_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # need SMA30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma10[i]) or np.isnan(sma30[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma10_1w_aligned[i]) or 
            np.isnan(sma30_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below SMA10 or 1w trend turns bearish
            if close[i] < sma10[i] or sma10_1w_aligned[i] < sma30_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above SMA10 or 1w trend turns bullish
            if close[i] > sma10[i] or sma10_1w_aligned[i] > sma30_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price > SMA10 > SMA30 and 1w SMA10 > SMA30 with volume surge
            if (close[i] > sma10[i] > sma30[i] and 
                sma10_1w_aligned[i] > sma30_1w_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price < SMA10 < SMA30 and 1w SMA10 < SMA30 with volume surge
            elif (close[i] < sma10[i] < sma30[i] and 
                  sma10_1w_aligned[i] < sma30_1w_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals