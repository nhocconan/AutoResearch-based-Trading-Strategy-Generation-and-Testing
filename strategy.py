#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Weekly EMA20 Trend and Volume Confirmation.
Long: Price breaks above 20-week Donchian upper channel + price > weekly EMA20 + volume > 2x 20-period average.
Short: Price breaks below 20-week Donchian lower channel + price < weekly EMA20 + volume > 2x 20-period average.
Exit: Price returns to weekly midline (mean reversion) or trend reverses.
Designed for 1d timeframe: targets 30-100 total trades over 4 years (7-25/year).
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
    
    # Get weekly data for Donchian channels and EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 20-week Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = np.full(len(high_1w), np.nan, dtype=np.float64)
    donchian_low_20 = np.full(len(low_1w), np.nan, dtype=np.float64)
    for i in range(19, len(high_1w)):
        donchian_high_20[i] = np.max(high_1w[i-19:i+1])
        donchian_low_20[i] = np.min(low_1w[i-19:i+1])
    
    # 20-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly midline for mean reversion exit
    weekly_midline = (donchian_high_20 + donchian_low_20) / 2.0
    weekly_midline_aligned = align_htf_to_ltf(prices, df_1w, weekly_midline)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Donchian (20 periods), weekly EMA (20 periods), volume MA (20 periods)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(weekly_midline_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        donchian_high = donchian_high_20_aligned[i]
        donchian_low = donchian_low_20_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        midline = weekly_midline_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish trend + volume spike
            if price > donchian_high and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + bearish trend + volume spike
            elif price < donchian_low and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midline (mean reversion) or trend turns bearish
            if price <= midline or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midline (mean reversion) or trend turns bullish
            if price >= midline or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Donchian_Breakout_WeeklyEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0