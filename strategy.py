#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeFilter
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter and volume confirmation work in both bull and bear markets.
Enter long when price breaks above 20-day high AND weekly close > weekly EMA20 (uptrend) AND volume > 1.5 * 20-day average volume.
Enter short when price breaks below 20-day low AND weekly close < weekly EMA20 (downtrend) AND volume > 1.5 * 20-day average volume.
Exit when price returns to 20-day midpoint OR weekly trend reverses.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 trades over 4 years (7-25/year).
Donchian channels provide objective breakout levels; weekly trend filter ensures alignment with higher timeframe structure.
Volume confirmation filters weak breakouts. Designed for BTC/ETH with SOL as secondary.
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 20-day Donchian channels (using daily high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 20-day Donchian (20), weekly EMA20 (20), volume avg (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        weekly_ema = ema_20_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Donchian levels with weekly trend filter AND volume
            # Long: price breaks above upper band AND weekly uptrend AND volume
            long_condition = (high_val > upper) and (close_val > weekly_ema) and vol_conf
            # Short: price breaks below lower band AND weekly downtrend AND volume
            short_condition = (low_val < lower) and (close_val < weekly_ema) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to midpoint OR weekly trend breaks down
            exit_condition = (close_val <= mid) or (close_val < weekly_ema)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to midpoint OR weekly trend breaks up
            exit_condition = (close_val >= mid) or (close_val > weekly_ema)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0