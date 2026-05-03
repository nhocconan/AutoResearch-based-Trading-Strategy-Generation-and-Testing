#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversal with 1d volume spike and chop regime filter.
# Williams %R identifies overbought/oversold conditions. Extreme readings below -90 (oversold) or above -10 (overbought) 
# signal potential reversals. Entry requires: Williams %R crossing back above -90 from below for longs, 
# or crossing below -10 from above for shorts, confirmed by 1d volume spike (>2x 20-period MA) and 
# choppiness regime (CHOP > 61.8 = ranging market favorable for mean reversion). 
# Exit when Williams %R returns to neutral zone (-50) or chop regime ends (CHOP < 38.2).
# Designed to work in both bull (buy oversold bounces) and bear (sell overbought bounces) markets 
# by fading extremes in ranging conditions. Target: 20-50 trades/year.

name = "4h_WilliamsR_Extreme_1dVolumeSpike_ChopRegime"
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
    
    # Calculate 4h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_4h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_4h = ((highest_high_4h - close) / (highest_high_4h - lowest_low_4h)) * -100
    # Handle division by zero when high == low
    williams_r_4h = np.where((highest_high_4h - lowest_low_4h) == 0, -50, williams_r_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_4h[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 2x 20-period volume MA
        # Need to get current 1d volume - use the aligned 1d data point
        # Since we're on 4h timeframe, we need to check if current 4h bar is within 1d bar
        # Simpler approach: use 1d volume directly from df_1d aligned
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        volume_spike = vol_1d_aligned[i] > (vol_ma_20_1d_aligned[i] * 2.0)
        
        # Chop regime condition: CHOP > 61.8 = ranging (good for mean reversion)
        chop_ranging = chop_1d_aligned[i] > 61.8
        chop_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: Williams %R crossed above -90 from below (exiting oversold) AND volume spike AND chop ranging AND session
            if i > 0 and williams_r_4h[i-1] <= -90 and williams_r_4h[i] > -90 and volume_spike and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crossed below -10 from above (exiting overbought) AND volume spike AND chop ranging AND session
            elif i > 0 and williams_r_4h[i-1] >= -10 and williams_r_4h[i] < -10 and volume_spike and chop_ranging:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) OR chop regime ends (trending) OR reverse signal
            if williams_r_4h[i] >= -50 or chop_trending or (i > 0 and williams_r_4h[i-1] >= -10 and williams_r_4h[i] < -10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) OR chop regime ends (trending) OR reverse signal
            if williams_r_4h[i] <= -50 or chop_trending or (i > 0 and williams_r_4h[i-1] <= -90 and williams_r_4h[i] > -90):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals