#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1d weekly pivot R1 AND volume > 2.0x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1d weekly pivot S1 AND volume > 2.0x 20-bar avg
# Exit when price retests Donchian midline (average of upper/lower bands)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to avoid overtrading.
# Weekly pivot provides structural HTF bias (bullish/bearish) from higher timeframe.
# Donchian breakout captures momentum; volume confirmation filters weak breakouts.
# Works in bull markets by capturing breakouts aligned with weekly pivot bias.
# Works in bear markets by shorting breakdowns aligned with weekly pivot bias.
# Weekly pivot avoids look-ahead by using prior week's data.

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot and Donchian calculation (using prior week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter (alternative to weekly pivot if needed)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from prior 1d data (using typical pivot formula)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    # We approximate using rolling window on 1d data (5 trading days ≈ 1 week)
    window = 5
    if len(high_1d) >= window:
        weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
        weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
        weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
        
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        weekly_r1 = weekly_pivot + weekly_range * 1.0 / 4.0  # R1 level
        weekly_s1 = weekly_pivot - weekly_range * 1.0 / 4.0  # S1 level
        
        # Align weekly pivot levels to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        # Not enough data for weekly pivot
        return np.zeros(n)
    
    # Calculate Donchian channels on 6h data
    window_dc = 20
    donchian_upper = pd.Series(high).rolling(window=window_dc, min_periods=window_dc).max().values
    donchian_lower = pd.Series(low).rolling(window=window_dc, min_periods=window_dc).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, window_dc)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_pivot = weekly_pivot_aligned[i]
        curr_r1 = weekly_r1_aligned[i]
        curr_s1 = weekly_s1_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midline
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midline
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > weekly R1 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_r1 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < weekly S1 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_s1 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals