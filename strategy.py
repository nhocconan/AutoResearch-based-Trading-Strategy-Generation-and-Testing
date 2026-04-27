#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-day volume confirmation and 1-week trend filter.
Trades when price crosses above/below Alligator teeth (median line) with volume > 1.5x weekly average
and weekly EMA confirms direction. Designed for low-frequency, high-conviction trades in both bull
and bear markets by using weekly trend filter and volume spike confirmation. Target: 15-30 trades/year
per symbol (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close):
    """Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shift"""
    # SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    # Shift forward by future values (as per Williams)
    jaw = jaw.shift(8)
    teeth = teeth.shift(5)
    lips = lips.shift(3)
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12-hour data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    jaw_12h, teeth_12h, lips_12h = williams_alligator(high_12h, low_12h, close_12h)
    
    # Align Alligator lines to 12-hour timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Alligator lines, volume MA, and weekly EMA
    start_idx = max(13, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 12-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_20_1w_aligned[i]
        
        # Current Alligator lines
        teeth_now = teeth_12h_aligned[i]
        lips_now = lips_12h_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Alligator crossover with volume and weekly trend alignment
        if position == 0:
            # Long: price crosses above teeth with volume + weekly uptrend
            if price_now > teeth_now and lips_now <= teeth_now and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price crosses below teeth with volume + weekly downtrend
            elif price_now < teeth_now and lips_now >= teeth_now and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lips or weekly trend turns down
            if price_now < lips_now or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above lips or weekly trend turns up
            if price_now > lips_now or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0