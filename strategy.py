#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-day volume confirmation and 1-week trend filter.
Trades when price crosses above/below Alligator jaws (SMMA13) with volume > 1.5x daily average
and weekly EMA34 confirms trend direction. Designed for low-frequency, high-conviction trades
to minimize fee drag in both bull and bear markets.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    res = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    res[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(arr)):
        res[i] = (res[i-1] * (period-1) + arr[i]) / period
    return res

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily timeframe
    # Jaws (Blue): SMMA(13), Teeth (Red): SMMA(8), Lips (Green): SMMA(5)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Typical price for Alligator calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    jaws = smma(typical_price_1d, 13)  # SMMA(13)
    teeth = smma(typical_price_1d, 8)   # SMMA(8)
    lips = smma(typical_price_1d, 5)    # SMMA(5)
    
    # Align Alligator lines to 12-hour timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily volume MA(20) for volume filter
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Alligator lines, weekly EMA, and daily volume MA
    start_idx = max(34, 34, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
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
        trend_1w = ema_34_1w_aligned[i]
        
        # Current Alligator values
        jaws_now = jaws_aligned[i]
        teeth_now = teeth_aligned[i]
        lips_now = lips_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Alligator alignment check: lips > teeth > jaws (bullish) or lips < teeth < jaws (bearish)
        bullish_align = lips_now > teeth_now and teeth_now > jaws_now
        bearish_align = lips_now < teeth_now and teeth_now < jaws_now
        
        # Entry conditions: Alligator crossover with volume and weekly trend alignment
        if position == 0:
            # Long: price crosses above jaws with bullish alignment + volume + weekly uptrend
            if (price_now > jaws_now and 
                lips_now <= jaws_now and  # previous lips was at or below jaws (crossing up)
                bullish_align and 
                vol_filter and 
                price_now > trend_1w):
                signals[i] = size
                position = 1
            # Short: price crosses below jaws with bearish alignment + volume + weekly downtrend
            elif (price_now < jaws_now and 
                  lips_now >= jaws_now and  # previous lips was at or above jaws (crossing down)
                  bearish_align and 
                  vol_filter and 
                  price_now < trend_1w):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth or weekly trend turns down
            if price_now < teeth_now or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above teeth or weekly trend turns up
            if price_now > teeth_now or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0