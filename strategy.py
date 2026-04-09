#!/usr/bin/env python3
# 12h_donchian_1d_volume_chop_v3
# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Uses 12h timeframe for low trade frequency (12-37/year target). Donchian breakout captures trends with clear structure,
# 1d volume spike confirms institutional participation, choppiness filter avoids whipsaw in ranging markets.
# Works in bull/bear markets: Donchian adapts to volatility, volume confirms breakout strength, chop filter improves win rate in ranges.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1d_volume_chop_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: highest high over 20 periods
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over 20 periods
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 12h timeframe (completed 12h candle only)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Get 1d HTF data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 1.5)
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_h = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_h - min_l
    
    # Choppiness Index: 100 * log10(tr_sum / range_14) / log10(14)
    chop = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe (completed daily candle only)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h lower Donchian
            if close[i] < lower_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h upper Donchian
            if close[i] > upper_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 12h upper Donchian, with volume spike, in trending regime (chop < 61.8)
            if (close[i] > upper_12h_aligned[i]) and vol_spike_aligned[i] and (chop_aligned[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h lower Donchian, with volume spike, in trending regime (chop < 61.8)
            elif (close[i] < lower_12h_aligned[i]) and vol_spike_aligned[i] and (chop_aligned[i] < 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals