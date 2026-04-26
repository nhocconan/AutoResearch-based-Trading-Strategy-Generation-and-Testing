#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_WeeklyTrend
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts in direction of weekly trend (price vs weekly EMA50), with entry confirmed by proximity to weekly Camarilla S3/R3 (mean reversion zones) and volume spike. Weekly trend filter avoids counter-trend trades in bear markets, while Camarilla levels provide favorable entry points within the trend. Volume spike confirms breakout strength. Designed for low trade frequency (12-37/year) to minimize fee drag, works in both bull/bear via weekly trend alignment.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly Camarilla levels (R3, S3) from prior weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    rng = high_1w - low_1w
    r3 = close_1w + 1.125 * rng
    s3 = close_1w - 1.125 * rng
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily data for volume median (more stable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    vol_median_20_1d = volume_1d_series.rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # 6h Donchian(20) - using 6h high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 6h volume > 1.5x daily median volume (scaled)
    volume_spike = volume > (1.5 * vol_median_20_1d_aligned)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for weekly EMA, 20 for Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_median_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_trend_up = close_val > ema_50_1w_aligned[i]
        weekly_trend_down = close_val < ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above Donchian high, in weekly uptrend, near S3 (support), volume spike
            long_cond = (close_val > donchian_high[i]) and weekly_trend_up and (close_val <= s3_aligned[i] * 1.02) and vol_spike
            # Short: price breaks below Donchian low, in weekly downtrend, near R3 (resistance), volume spike
            short_cond = (close_val < donchian_low[i]) and weekly_trend_down and (close_val >= r3_aligned[i] * 0.98) and vol_spike
            
            if long_cond:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_cond:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on weekly trend reversal or at Donchian high (trailing)
            if close_val < ema_50_1w_aligned[i] or close_val < donchian_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on weekly trend reversal or at Donchian low (trailing)
            if close_val > ema_50_1w_aligned[i] or close_val > donchian_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0