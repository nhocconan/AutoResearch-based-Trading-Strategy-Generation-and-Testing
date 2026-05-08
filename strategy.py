#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with volume spike and choppiness regime filter.
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Entry when Williams %R crosses below -80 (oversold) or above -20 (overbought) with volume confirmation.
# Choppiness index (14) > 61.8 indicates ranging market for mean reversion trades.
# Designed for low trade frequency (20-30/year) to avoid fee drag. Works in both trending and ranging markets.

name = "4h_1dWilliamsR_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Williams %R (14-period)
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    wr = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:
            wr[i] = -100 * (hh - close_1d[i]) / (hh - ll)
        else:
            wr[i] = -50  # Avoid division by zero
    
    # Calculate Choppiness Index (14-period)
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                 abs(high_1d[i] - close_1d[i-1]), 
                 abs(low_1d[i] - close_1d[i-1]))
        tr_1d[i] = tr
    
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        sum_tr = np.sum(tr_1d[i-13:i+1])
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align 1d indicators to 4h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 4h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Williams %R and Chop
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(wr_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses below -80 (oversold) in ranging market with volume
            if (wr_aligned[i] < -80 and 
                chop_aligned[i] > 61.8 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses above -20 (overbought) in ranging market with volume
            elif (wr_aligned[i] > -20 and 
                  chop_aligned[i] > 61.8 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or chop drops below 38.2 (trending)
            if wr_aligned[i] > -50 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or chop drops below 38.2 (trending)
            if wr_aligned[i] < -50 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals