#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d Choppiness regime filter
# Long when price breaks above Donchian high(20) + volume spike + 1d chop > 61.8 (range)
# Short when price breaks below Donchian low(20) + volume spike + 1d chop > 61.8 (range)
# Exit when price crosses Donchian midline (20-period average of high/low)
# Works in both bull and bear: breakouts with volume in ranging markets (chop > 61.8) capture reversals
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Choppiness Index for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first element NaN
    
    # ATR(14)
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.nanmean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(atr)
    for i in range(len(sum_atr)):
        if i < 14:
            sum_atr[i] = np.nan
        else:
            sum_atr[i] = np.nansum(atr[i-13:i+1])
    
    # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.nanmax(high_1d[i-13:i+1])
            min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = 100 * (np.log10(sum_atr) - np.log10(max_high - min_low)) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period)
    donch_high = np.zeros_like(close)
    donch_low = np.zeros_like(close)
    for i in range(len(close)):
        if i < 20:
            donch_high[i] = np.nan
            donch_low[i] = np.nan
        else:
            donch_high[i] = np.nanmax(high[i-20:i+1])
            donch_low[i] = np.nanmin(low[i-20:i+1])
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_median[i] = np.nan
        else:
            vol_median[i] = np.nanmedian(volume[i-20:i+1])
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Chop > 61.8 indicates ranging market (good for mean reversion breakouts)
        in_range = chop_aligned[i] > 61.8
        
        # Long: price breaks above Donchian high + volume spike + ranging market
        if (close[i] > donch_high[i] and 
            volume[i] > vol_threshold[i] and 
            in_range):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume spike + ranging market
        elif (close[i] < donch_low[i] and 
              volume[i] > vol_threshold[i] and 
              in_range):
            signals[i] = -0.25
        
        # Exit: price crosses Donchian midline (average of high/low)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donch_high[i] + donch_low[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donch_high[i] + donch_low[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0