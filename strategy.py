#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter.
# Donchian breakout captures momentum. Entry requires price breaking above/below 20-period Donchian channel
# confirmed by 1d volume spike (>1.5x 20-period volume MA) and chop regime (CHOP > 61.8 = ranging, favorable for breakout continuation).
# Exit on opposite Donchian(10) touch or chop regime ending (CHOP < 38.2 = trending, risk of false breakout).
# Works in bull/bear by following momentum in ranging markets where breakouts have higher success rate.
# Target: 25-40 trades/year.

name = "4h_Donchian20_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d choppiness index (CHOP) - ranging market indicator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14-period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: CHOP = 100 * log10( sum(ATR14) / (HH14 - LL14) ) / log10(14)
    # Avoid division by zero
    range_1d = hh_1d - ll_1d
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)  # small epsilon to prevent div by zero
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(atr_sum_1d / range_1d) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    # Calculate 4h Donchian channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 1.5x 20-period volume MA
        volume_spike = vol_1d_aligned[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        
        # Chop regime condition: CHOP > 61.8 = ranging (favorable for breakout)
        chop_ranging = chop_1d_aligned[i] > 61.8
        chop_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above 20-period Donchian high AND volume spike AND chop ranging AND session
            if close[i] > highest_high_20[i] and volume_spike and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period Donchian low AND volume spike AND chop ranging AND session
            elif close[i] < lowest_low_20[i] and volume_spike and chop_ranging:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 10-period Donchian low OR chop regime ends (trending) OR reverse signal
            if close[i] < lowest_low_10[i] or chop_trending or (close[i] < lowest_low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 10-period Donchian high OR chop regime ends (trending) OR reverse signal
            if close[i] > highest_high_10[i] or chop_trending or (close[i] > highest_high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals