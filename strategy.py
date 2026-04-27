#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with daily trend filter and volume confirmation.
Breakout above 20-period high in uptrend: long. Breakdown below 20-period low in downtrend: short.
Uses daily EMA34 for trend filter and daily volume spike for confirmation. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    d_close = df_1d['close'].values
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily volume average for confirmation
    d_volume = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(d_volume).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + EMA (34) + volume average (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_avg = vol_avg_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Breakout above Donchian high in uptrend: long
            if price_now > donchian_high and price_now > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Breakdown below Donchian low in downtrend: short
            elif price_now < donchian_low and price_now < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below Donchian low or trend change
            if price_now < donchian_low or price_now < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above Donchian high or trend change
            if price_now > donchian_high or price_now > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0