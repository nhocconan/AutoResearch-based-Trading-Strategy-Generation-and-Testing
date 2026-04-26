#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter_Volume
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, choppiness regime (CHOP>61.8 = range), and volume confirmation on 12h timeframe.
Only trade breakouts in trending 1d market (close > EMA34 for long, close < EMA34 for short) when market is not choppy (CHOP <= 61.8) and volume > 1.5x 20-period median.
Uses discrete sizing 0.25 to target 12-37 trades/year. Designed to work in bull/bear via 1d trend filter and avoid whipsaws via regime/volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3, R4, S4
    rng = high_1d - low_1d
    r3 = close_1d + 1.125 * rng
    s3 = close_1d - 1.125 * rng
    r4 = close_1d + 1.5 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    # Using 14-period CHOP on 1d data
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first period TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 14*2 for CHOP, 20 for volume median
    start_idx = max(34, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i]) or
            np.isnan(r4_12h[i]) or
            np.isnan(s4_12h[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Only trade when market is NOT choppy (CHOP <= 61.8 indicates trending)
            not_choppy = chop_val <= 61.8
            
            # Long: price > R3 and volume spike, in uptrend (close > EMA34), not choppy
            long_entry = (close_val > r3_12h[i]) and vol_spike and (close_val > ema_34_val) and not_choppy
            # Short: price < S3 and volume spike, in downtrend (close < EMA34), not choppy
            short_entry = (close_val < s3_12h[i]) and vol_spike and (close_val < ema_34_val) and not_choppy
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or at R4 (take profit)
            if close_val < ema_34_val or close_val > r4_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S4 (take profit)
            if close_val > ema_34_val or close_val < s4_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter_Volume"
timeframe = "12h"
leverage = 1.0