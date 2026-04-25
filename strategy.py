#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: 6h Donchian breakouts aligned with weekly pivot direction (from 1w data)
and volume spikes capture institutional flow with reduced false signals.
Weekly pivot provides longer-term bias that works in both bull and bear markets.
Target: 12-37 trades/year on 6h (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation (primary timeframe HTF for structure)
    df_6h = get_htf_data(prices, '6h')
    # Get 1d data for volume average calculation
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_6h) < 30 or len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Upper = max(high, 20), Lower = min(low, 20)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (no extra delay - based on completed 6h bar)
    donchian_upper_6h = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_6h = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Calculate 1d volume average (20-period) for volume spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly pivot points (standard formula) from 1w data
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Weekly bias: price above pivot = bullish bias, below = bearish bias
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    
    # Align weekly bias and pivot levels to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_6h[i]) or 
            np.isnan(donchian_lower_6h[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_upper_val = donchian_upper_6h[i]
        donchian_lower_val = donchian_lower_6h[i]
        vol_ma_20_val = vol_ma_20_1d_aligned[i]
        weekly_bullish_val = weekly_bullish_aligned[i] > 0.5
        weekly_bearish_val = weekly_bearish_aligned[i] > 0.5
        r3_val = r3_1w_aligned[i]
        s3_val = s3_1w_aligned[i]
        
        # Volume spike: current 6h volume > 2.0 * 20-period 1d volume average
        # (Using 1d volume average as proxy for normal 6h volume)
        volume_spike = curr_volume > 2.0 * vol_ma_20_val
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly bullish bias AND volume spike
            long_condition = (curr_high > donchian_upper_val) and weekly_bullish_val and volume_spike
            # Short: price breaks below Donchian lower AND weekly bearish bias AND volume spike
            short_condition = (curr_low < donchian_lower_val) and weekly_bearish_val and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR approximation using price range) or weekly bias change
            # Simple ATR approximation: use 6-bar range
            if i >= 6:
                approx_atr = np.max(high[i-5:i+1]) - np.min(low[i-5:i+1])
            else:
                approx_atr = np.max(high[:i+1]) - np.min(low[:i+1])
            
            if curr_close <= entry_price - 2.0 * approx_atr or not weekly_bullish_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss or weekly bias change
            if i >= 6:
                approx_atr = np.max(high[i-5:i+1]) - np.min(low[i-5:i+1])
            else:
                approx_atr = np.max(high[:i+1]) - np.min(low[:i+1])
            
            if curr_close >= entry_price + 2.0 * approx_atr or not weekly_bearish_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0