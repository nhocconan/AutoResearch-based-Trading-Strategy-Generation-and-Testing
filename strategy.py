#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Donchian breakouts capture strong momentum moves with clear structure
# 1d volume > 1.5x 20-period average confirms institutional participation
# Chop index > 61.8 ensures we only trade in ranging markets (mean reversion after false breakouts)
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_1dVolumeChop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Chop index for regime filter (range = chop > 61.8)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14) - Wilder's smoothing
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop index = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    sum_tr_14 = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        sum_tr_14[13] = np.sum(tr[:14])
        for i in range(14, len(tr)):
            sum_tr_14[i] = sum_tr_14[i-1] - tr[i-14] + tr[i]
    
    max_high_14 = np.full_like(high_1d, np.nan, dtype=float)
    min_low_14 = np.full_like(low_1d, np.nan, dtype=float)
    if len(high_1d) >= 14:
        for i in range(13, len(high_1d)):
            max_high_14[i] = np.max(high_1d[i-13:i+1])
            min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    chop = np.full_like(tr, np.nan, dtype=float)
    valid = (~np.isnan(sum_tr_14)) & (~np.isnan(max_high_14)) & (~np.isnan(min_low_14)) & ((max_high_14 - min_low_14) > 0)
    chop[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
    
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_period = 20
    highest_high = np.full_like(high, np.nan, dtype=float)
    lowest_low = np.full_like(low, np.nan, dtype=float)
    
    if len(high) >= donchian_period:
        for i in range(donchian_period-1, len(high)):
            highest_high[i] = np.max(high[i-donchian_period+1:i+1])
            lowest_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
            
        # Regime filter: only trade when chop > 61.8 (ranging market)
        if chop_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_12h = volume[i]
        curr_donchian_high = highest_high[i]
        curr_donchian_low = lowest_low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_ma_1d = vol_ma_20_aligned[i]
        curr_volume_1d = df_1d['volume'].values[min(i // 2, len(df_1d)-1)] if i // 2 < len(df_1d) else df_1d['volume'].values[-1]
        
        if position == 0:  # Flat - look for mean reversion entries
            # Require 1d volume confirmation
            if curr_volume_1d > (1.5 * curr_vol_ma_1d):
                # Long entry: price near Donchian low (oversold in range)
                if curr_close <= curr_donchian_low * 1.002:  # within 0.2% of low
                    signals[i] = 0.25
                    position = 1
                # Short entry: price near Donchian high (overbought in range)
                elif curr_close >= curr_donchian_high * 0.998:  # within 0.2% of high
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price reaches midpoint of Donchian channel (mean reversion target)
            midpoint = (curr_donchian_high + curr_donchian_low) / 2
            if curr_close >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price reaches midpoint of Donchian channel
            midpoint = (curr_donchian_high + curr_donchian_low) / 2
            if curr_close <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals