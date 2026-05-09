#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Choppiness Index regime filter and 1d Williams %R for mean-reversion entries.
# Uses 1d Choppiness Index > 61.8 to identify ranging markets, then fades extremes using Williams %R.
# Designed for low trade frequency (<25/year) to avoid fee drag in 4h timeframe.
# Works in both bull/bear markets by requiring ranging conditions and mean-reversion at extremes.
name = "4h_ChopWilliams_MeanRev"
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
    
    # Get 1d data for Choppiness Index and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR (14-period)
    atr = np.full(len(high_1d), np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    chop = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        if not np.isnan(atr[i-13:i+1]).any() and not np.isnan(high_1d[i-13:i+1]).any() and not np.isnan(low_1d[i-13:i+1]).any():
            sum_tr = np.nansum(atr[i-13:i+1])
            hh = np.nanmax(high_1d[i-13:i+1])
            ll = np.nanmin(low_1d[i-13:i+1])
            if hh > ll:
                chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    for i in range(14, len(high_1d)):
        highest_high[i] = np.nanmax(high_1d[i-13:i+1])
        lowest_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    williams_r = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        if highest_high[i] > lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Align 1d indicators to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    williams_r_4h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: spike above 2.0x 6-period average (1.5 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for Williams %R calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_4h[i]) or np.isnan(williams_r_4h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        # Chop > 61.8 indicates ranging market (good for mean reversion)
        chop_high = chop_4h[i] > 61.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in ranging market
            if williams_r_4h[i] < -80 and chop_high and vol_ok and in_session:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in ranging market
            elif williams_r_4h[i] > -20 and chop_high and vol_ok and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or chop breaks down
            if williams_r_4h[i] > -50 or chop_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or chop breaks down
            if williams_r_4h[i] < -50 or chop_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals