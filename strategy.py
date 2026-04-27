#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and Weekly Trend Filter.
Long when price breaks above Donchian(20) high AND price > weekly EMA20 AND volume > 1.5x average.
Short when price breaks below Donchian(20) low AND price < weekly EMA20 AND volume > 1.5x average.
Exit when price crosses back through Donchian(20) midpoint.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan, dtype=np.float64)
    donchian_low = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20) + weekly EMA + volume MA
    start_idx = max(19, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        midpoint = donchian_mid[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > weekly EMA20 AND volume spike
            if price_now > upper and price_now > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low AND price < weekly EMA20 AND volume spike
            elif price_now < lower and price_now < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if price_now < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price_now > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_WeeklyTrend"
timeframe = "4h"
leverage = 1.0