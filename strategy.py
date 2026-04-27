#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily Donchian(20) breakouts aligned with weekly trend filter and volume confirmation.
Enter long when price breaks above 20-day high AND weekly close > weekly EMA20 (uptrend) AND volume > 1.5 * 20-day average.
Enter short when price breaks below 20-day low AND weekly close < weekly EMA20 (downtrend) AND volume > 1.5 * 20-day average.
Exit when price returns to 20-day midpoint OR weekly trend reverses.
Designed for low-frequency, high-conviction trades (target: 30-80 total over 4 years) to minimize fee drag.
Uses 1w HTF for trend alignment to avoid whipsaws in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) channels (using previous 20 days, not including today)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20+1), weekly EMA (20), volume avg (20)
    start_idx = max(21, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        weekly_ema = ema_20_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly trend filter AND volume
            # Long: price breaks above upper band AND weekly uptrend AND volume
            long_condition = (close_val > upper) and (close_val > weekly_ema) and vol_conf
            # Short: price breaks below lower band AND weekly downtrend AND volume
            short_condition = (close_val < lower) and (close_val < weekly_ema) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to midpoint OR weekly trend breaks
            exit_condition = (close_val <= mid) or (close_val < weekly_ema)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to midpoint OR weekly trend breaks
            exit_condition = (close_val >= mid) or (close_val > weekly_ema)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0