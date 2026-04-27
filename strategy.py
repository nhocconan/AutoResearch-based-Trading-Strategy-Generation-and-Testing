#!/usr/bin/env python3
"""
Hypothesis: 6-hour strategy using 6-hour Donchian breakout with 1-day trend filter and volume confirmation.
Enters long when price breaks above 20-period Donchian upper band with above-average volume and daily uptrend.
Enters short when price breaks below 20-period Donchian lower band with above-average volume and daily downtrend.
Uses daily timeframe for trend filter and volume filter, 6-hour for entry timing.
Designed to work in both bull and bear markets by following the daily trend and requiring volume confirmation.
Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1-day volume MA(20) for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6-hour Donchian channels (20-period)
    high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, EMA, and volume MA
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Current Donchian levels
        upper = high_6h[i]
        lower = low_6h[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: 6h Donchian breakout with volume and daily trend alignment
        if position == 0:
            # Long: price breaks above upper band with volume + daily uptrend
            if price_now > upper and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + daily downtrend
            elif price_now < lower and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to mid-band or daily trend turns down
            mid_band = (upper + lower) / 2.0
            if price_now < mid_band or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to mid-band or daily trend turns up
            mid_band = (upper + lower) / 2.0
            if price_now > mid_band or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_DonchianBreakout_1dVolume_1dTrend"
timeframe = "6h"
leverage = 1.0