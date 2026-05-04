#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme + 1d volume spike + chop regime filter
# In trending markets (1d CHOP < 42): fade extreme %R reversals (mean reversion at extremes)
# In ranging markets (1d CHOP >= 42): trade %R reversals from overbought/oversold
# Volume confirmation (>1.8x 20-period EMA) ensures institutional participation
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
# Williams %R identifies exhaustion points that work in both bull and bear markets via regime filter.

name = "4h_WilliamsR_1dChop_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14 period) - momentum oscillator
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 is overbought, < -80 is oversold
    williams_r = -100 * (hh_14 - close_1d) / (hh_14 - ll_14)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan)  # handle division by zero
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    # True Range
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    tr_sum_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    hh_ll_diff = hh_14 - ll_14
    chop_1d = np.where(
        (hh_ll_diff > 0) & (~tr_sum_14.isna()) & (~hh_ll_diff.isna()),
        100 * np.log10(tr_sum_14 / hh_ll_diff) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 20-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            if chop_aligned[i] < 42:
                # Trending market: fade extreme Williams %R (mean reversion)
                # Long when oversold (< -80) and turning up
                if williams_r_aligned[i] < -80 and williams_r_aligned[i] > williams_r_aligned[i-1]:
                    if volume_confirm:
                        signals[i] = 0.25
                        position = 1
                # Short when overbought (> -20) and turning down
                elif williams_r_aligned[i] > -20 and williams_r_aligned[i] < williams_r_aligned[i-1]:
                    if volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: trade Williams %R reversals from extremes
                # Long when oversold (< -80) and volume confirmation
                if williams_r_aligned[i] < -80 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (> -20) and volume confirmation
                elif williams_r_aligned[i] > -20 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to midpoint (-50) OR chop increases (>50) OR volume drops
            if (williams_r_aligned[i] >= -50 or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to midpoint (-50) OR chop increases (>50) OR volume drops
            if (williams_r_aligned[i] <= -50 or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals