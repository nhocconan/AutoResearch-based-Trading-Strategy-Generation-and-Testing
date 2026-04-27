#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with 1-week trend filter and volume confirmation.
Trades breakouts of the 20-day Donchian channel when weekly trend confirms and volume exceeds 2x average.
Designed to capture momentum in both bull and bear markets by using weekly trend as filter and volume to confirm breakout strength.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-week EMA for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1-day volume MA(20) for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute day of week filter (Monday-Friday only)
    days = pd.DatetimeIndex(prices['open_time']).weekday  # Monday=0, Sunday=6
    
    # Warmup: need Donchian channels, weekly EMA, and volume MA
    start_idx = max(20, 50, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Day of week filter: only trade Monday-Friday (0-4)
        day = days[i]
        if day >= 5:  # Saturday=5, Sunday=6
            signals[i] = 0.0
            continue
        
        # Current daily price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Current Donchian levels
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        
        # Volume filter: volume > 2x 1-day average
        vol_filter = vol_now > 2.0 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and weekly trend alignment
        if position == 0:
            # Long: price breaks above Donchian high with volume + weekly uptrend
            if price_now > donch_high and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume + weekly downtrend
            elif price_now < donch_low and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retrace to midpoint or weekly trend turns down
            midpoint = (donch_high + donch_low) / 2.0
            if price_now < midpoint or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price retrace to midpoint or weekly trend turns up
            midpoint = (donch_high + donch_low) / 2.0
            if price_now > midpoint or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0