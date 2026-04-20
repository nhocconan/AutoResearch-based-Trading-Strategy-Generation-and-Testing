#!/usr/bin/env python3
# 4h_ThreeBarReversal_VolumeTrend
# Hypothesis: Three-bar reversal patterns (bullish/bearish) at key price levels (support/resistance)
# combined with volume confirmation and trend filter (ADX > 25) capture institutional reversals.
# Works in both bull and bear markets by identifying exhaustion points. Target: 20-50 trades/year.

name = "4h_ThreeBarReversal_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 3-bar reversal patterns
    # Bullish: 3 consecutive lower lows followed by a higher close
    # Bearish: 3 consecutive higher highs followed by a lower close
    bullish_rev = np.zeros(len(high_4h), dtype=bool)
    bearish_rev = np.zeros(len(high_4h), dtype=bool)
    
    for i in range(3, len(high_4h)):
        # Bullish reversal: lower lows for 3 bars, then higher close
        if (low_4h[i-3] > low_4h[i-2] > low_4h[i-1] and 
            close_4h[i] > close_4h[i-1]):
            bullish_rev[i] = True
        # Bearish reversal: higher highs for 3 bars, then lower close
        if (high_4h[i-3] < high_4h[i-2] < high_4h[i-1] and 
            close_4h[i] < close_4h[i-1]):
            bearish_rev[i] = True
    
    # Align reversal signals to LTF
    bullish_rev_aligned = align_htf_to_ltf(prices, df_4h, bullish_rev.astype(float))
    bearish_rev_aligned = align_htf_to_ltf(prices, df_4h, bearish_rev.astype(float))
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM (14-period)
    tr_sum = np.full_like(high_4h, np.nan)
    dm_plus_sum = np.full_like(high_4h, np.nan)
    dm_minus_sum = np.full_like(high_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 13:  # 14-period smoothing
            tr_sum[i] = np.nansum(tr[i-13:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-13:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-13:i+1])
    
    # Directional Indicators
    di_plus = np.full_like(high_4h, np.nan)
    di_minus = np.full_like(high_4h, np.nan)
    dx = np.full_like(high_4h, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(high_4h, np.nan)
    for i in range(len(high_4h)):
        if i >= 27:  # 14 + 13 for ADX smoothing
            valid_dx = dx[i-13:i+1]
            if not np.all(np.isnan(valid_dx)):
                adx[i] = np.nanmean(valid_dx)
    
    # Align ADX to LTF
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_4h > (vol_ma20 * 1.3)
    volume_filter_aligned = align_htf_to_ltf(prices, df_4h, volume_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 20)  # Ensure ADX and patterns are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_rev_aligned[i]) or np.isnan(bearish_rev_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish reversal + ADX > 25 + volume confirmation
            if bullish_rev_aligned[i] > 0.5 and adx_aligned[i] > 25 and volume_filter_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: bearish reversal + ADX > 25 + volume confirmation
            elif bearish_rev_aligned[i] > 0.5 and adx_aligned[i] > 25 and volume_filter_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on bearish reversal or ADX weakness
            if bearish_rev_aligned[i] > 0.5 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on bullish reversal or ADX weakness
            if bullish_rev_aligned[i] > 0.5 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals