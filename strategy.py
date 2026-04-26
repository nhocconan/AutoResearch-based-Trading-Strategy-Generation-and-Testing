#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPillar_VolumeRegime
Hypothesis: 6h Donchian(20) breakouts with 12h EMA200 trend filter and volume regime (top 30% volume).
Weekly pivot direction (from 1w data) filters breakout alignment: only long when price > weekly pivot,
only short when price < weekly pivot. This avoids counter-trend breakouts in ranging markets.
Volume regime ensures trades occur during high participation. Fixed size 0.25 to limit fees.
Target: 12-25 trades/year (~50-100 over 4 years) to stay within fee drag limits.
Works in bull (breakouts with trend) and bear (avoids false breakouts via weekly pivot filter).
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
    
    # Load 12h data ONCE before loop for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for weekly pivot calculation (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA200 for trend filter (long only above EMA, short only below EMA)
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Weekly pivot from 1d data: (prior week's high + low + close) / 3
    # We approximate weekly pivot using rolling window of 5 days (1 week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot: (max(high_5d) + min(low_5d) + close_1d) / 3
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_pivot = (high_5d + low_5d + close_1d) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian(20) breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (200 for EMA, 20 for Donchian, 50 for volume percentile)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_percentile_70[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema_200_val = ema_200_12h_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_regime = volume_regime[i]
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume regime, 12h EMA200 trend, and weekly pivot filter
        long_entry = (close_val > donchian_high_val) and vol_regime and (close_val > ema_200_val) and (close_val > weekly_pivot_val)
        short_entry = (close_val < donchian_low_val) and vol_regime and (close_val < ema_200_val) and (close_val < weekly_pivot_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian mean reversion (break of midpoint)
            mid_point = (donchian_high_val + donchian_low_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian mean reversion (break of midpoint)
            mid_point = (donchian_high_val + donchian_low_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPillar_VolumeRegime"
timeframe = "6h"
leverage = 1.0