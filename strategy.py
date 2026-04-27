#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Weekly EMA Trend and Volume Confirmation.
Long when: 1) Price breaks above weekly Donchian high (20-period), 2) Price > weekly EMA20 (bullish trend), 3) Volume > 2x 20-period average.
Short when: 1) Price breaks below weekly Donchian low (20-period), 2) Price < weekly EMA20 (bearish trend), 3) Volume > 2x 20-period average.
Exit when price returns to weekly midpoint (mean reversion) or trend reverses.
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
    
    # Get weekly data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = np.full(len(high_1w), np.nan, dtype=np.float64)
    donchian_low = np.full(len(low_1w), np.nan, dtype=np.float64)
    for i in range(19, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-19:i+1])
        donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + bullish trend + volume spike
            if price > donch_high and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low + bearish trend + volume spike
            elif price < donch_low and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly midpoint (mean reversion) or trend turns bearish
            weekly_mid = (donch_high + donch_low) / 2.0
            if price <= weekly_mid or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly midpoint (mean reversion) or trend turns bullish
            weekly_mid = (donch_high + donch_low) / 2.0
            if price >= weekly_mid or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Donchian_Breakout_WeeklyEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0