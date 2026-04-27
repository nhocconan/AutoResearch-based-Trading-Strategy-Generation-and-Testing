#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolSpike
Hypothesis: Uses 6h Donchian(20) breakouts with 12h EMA50 trend filter and 1d volume spike confirmation.
Long when price breaks above Donchian upper band AND 12h close > EMA50 (uptrend) AND 1d volume > 2.0 * 20-period average.
Short when price breaks below Donchian lower band AND 12h close < EMA50 (downtrend) AND 1d volume > 2.0 * 20-period average.
Exit when price returns to the 6h Donchian midpoint (mean reversion) OR trend reverses.
Designed for 6h timeframe to achieve 50-150 total trades over 4 years with low fee drag.
Works in both bull and bear markets by following 12h trend while using Donchian breakouts for precise entries.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d_series = pd.Series(df_1d['volume'].values)
    vol_avg_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), 12h EMA50 (50), 1d vol avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        ema_val = ema_50_12h_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_current = volume[i]
        vol_conf = vol_current > (2.0 * vol_avg)
        
        if position == 0:
            # Look for entry: breakout of Donchian bands with 12h trend filter AND 1d volume spike
            # Long: price breaks above upper band AND 12h uptrend AND volume spike
            long_condition = (close_val > upper) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower band AND 12h downtrend AND volume spike
            short_condition = (close_val < lower) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to Donchian midpoint OR trend breaks
            exit_condition = (close_val <= mid) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to Donchian midpoint OR trend breaks
            exit_condition = (close_val >= mid) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_1dVolSpike"
timeframe = "6h"
leverage = 1.0