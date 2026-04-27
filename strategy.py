#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian breakout with 12-hour volume confirmation and 1-day trend filter.
Trades breakouts of the 20-bar Donchian channel when volume exceeds 12-hour average and daily trend confirms.
Designed to work in both bull and bear markets by using daily trend as filter and volume to confirm breakout strength.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6-hour data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6-hour Donchian channel (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper and lower bands
    upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 6-hour timeframe
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    
    # Get 12-hour data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1-day EMA(25) for trend
    close_1d = df_1d['close'].values
    ema_25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian bands, volume MA, and daily EMA
    start_idx = max(20, 20, 25)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(ema_25_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 6-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_1d = ema_25_1d_aligned[i]
        
        # Current Donchian bands
        upper_now = upper_6h_aligned[i]
        lower_now = lower_6h_aligned[i]
        
        # Volume filter: volume > 1.5x 12-hour average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and daily trend alignment
        if position == 0:
            # Long: price breaks above upper band with volume + daily uptrend
            if price_now > upper_now and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + daily downtrend
            elif price_now < lower_now and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or daily trend turns down
            midpoint = (upper_now + lower_now) / 2
            if price_now < midpoint or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint or daily trend turns up
            midpoint = (upper_now + lower_now) / 2
            if price_now > midpoint or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_DonchianBreakout_12hVolume_1dTrend"
timeframe = "6h"
leverage = 1.0