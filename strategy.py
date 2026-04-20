#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d Pivot Point breakout + volume confirmation + 1d Choppiness regime filter.
# Long when price breaks above R1 pivot with volume > 1.5x average and market is trending (CHOP < 38.2).
# Short when price breaks below S1 pivot with volume > 1.5x average and market is trending (CHOP < 38.2).
# Uses daily Choppiness Index to avoid whipsaws in ranging markets.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's pivot points (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d Choppiness Index (14-period)
    atr_14 = np.zeros(len(high_1d))
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_high_low = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low / (atr_14 * 14)) / np.log10(14)
    
    # Align Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        chop_val = chop_aligned[i]
        vol_ok = vol_filter[i]
        
        # Trending market condition (CHOP < 38.2)
        trending = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above R1, trending market, volume
            if price > r1_val and trending and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, trending market, volume
            elif price < s1_val and trending and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or market becomes ranging
            if price < s1_val or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or market becomes ranging
            if price > r1_val or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_Breakout_Volume_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0