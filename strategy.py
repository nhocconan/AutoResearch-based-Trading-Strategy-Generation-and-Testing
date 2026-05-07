#!/usr/bin/env python3
name = "6H_Donchian_20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for volume confirmation (optional, but we can use it for context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    prev_week_high = df_1w['high'].values
    prev_week_low = df_1w['low'].values
    prev_week_close = df_1w['close'].values
    prev_week_open = df_1w['open'].values
    
    # Weekly Pivot Point (PP)
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Resistance and Support levels
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pp - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pp)
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily average volume for confirmation (20-period)
    vol_ma_daily = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_daily)
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
    # Since we're comparing 6h volume to daily average, we scale appropriately
    # Approximate: 6h volume should be > (1.5 * daily_avg_vol / 4) assuming 4x 6h periods per day
    vol_threshold = 1.5 * vol_ma_daily_aligned / 4.0
    volume_filter = volume > vol_threshold
    
    # Trend filter: price above/below weekly pivot
    # We use the weekly pivot as a bias indicator
    bias_long = close > pp_aligned
    bias_short = close < pp_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have Donchian and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(vol_ma_daily_aligned[i]) or vol_ma_daily_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above resistance with bullish bias and volume
            if (close[i] > donchian_high[i] and 
                bias_long[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below support with bearish bias and volume
            elif (close[i] < donchian_low[i] and 
                  bias_short[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Donchian breakout in opposite direction or loss of bias
            # For long: exit on Donchian breakdown or bearish bias
            # For short: exit on Donchian breakout or bullish bias
            if position == 1:
                if (close[i] < donchian_low[i] or not bias_long[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > donchian_high[i] or not bias_short[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals