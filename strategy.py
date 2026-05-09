#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian breakout with volume confirmation
# Works in bull (breakouts) and bear (mean reversion in chop)
# Limited trades via chop filter (avoid whipsaw) and volume confirmation
name = "4h_Choppiness_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index on daily timeframe
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # where ATR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    atr_list = []
    prev_close = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - prev_close)
    tr3 = abs(df_1d['low'] - prev_close)
    atr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max high and min low over 14 periods
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Donchian channels
    upper_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Volume filter: above 1.5x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        chop_value = chop_aligned[i]
        
        # Determine regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        if position == 0:
            # In trending regime: Donchian breakout
            if is_trending and vol_ok:
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                elif close[i] < lower_aligned[i]:
                    signals[i] = -0.30
                    position = -1
            # In ranging regime: mean reversion at Donchian bands
            elif is_ranging and vol_ok:
                if close[i] < lower_aligned[i]:
                    signals[i] = 0.30  # Buy at lower band
                    position = 1
                elif close[i] > upper_aligned[i]:
                    signals[i] = -0.30  # Sell at upper band
                    position = -1
        
        elif position == 1:
            # Exit long: opposite signal or chop extreme
            if (is_ranging and close[i] > upper_aligned[i]) or \
               (is_trending and close[i] < lower_aligned[i]) or \
               chop_value > 70:  # Extreme chop - exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: opposite signal or chop extreme
            if (is_ranging and close[i] < lower_aligned[i]) or \
               (is_trending and close[i] > upper_aligned[i]) or \
               chop_value > 70:  # Extreme chop - exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals