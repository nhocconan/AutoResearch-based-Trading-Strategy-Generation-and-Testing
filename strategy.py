#!/usr/bin/env python3
"""
12h Williams Alligator with 1d volume confirmation and 1d trend filter.
Trades when price crosses Alligator lines with volume above daily average and trend alignment.
Williams Alligator (13,8,5) smoothed with SMMA: Jaw(13,8), Teeth(8,5), Lips(5,3).
Long: Lips > Teeth > Jaw with volume confirmation and price > 1d EMA(50).
Short: Lips < Teeth < Jaw with volume confirmation and price < 1d EMA(50).
Target: 12-30 trades/year (48-120 total over 4 years) to minimize fee drag.
Works in bull/bear via 1d trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Jaw (13,8): SMMA of median price over 13 periods
    median_12h = (high_12h + low_12h) / 2
    jaw_raw = smma(median_12h, 13)
    
    # Teeth (8,5): SMMA of median price over 8 periods
    teeth_raw = smma(median_12h, 8)
    
    # Lips (5,3): SMMA of median price over 5 periods
    lips_raw = smma(median_12h, 5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_raw)
    
    # Get daily data for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (00-23 UTC - trade all hours for 12h)
    # For 12h timeframe, we can trade all hours as each bar represents half a day
    
    # Warmup: need Alligator lines, volume MA, and daily EMA
    start_idx = max(13, 8, 5, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Current Alligator values
        jaw_now = jaw_aligned[i]
        teeth_now = teeth_aligned[i]
        lips_now = lips_aligned[i]
        
        # Volume filter: volume > 1.2x 1-day average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Alligator alignment
        lips_above_teeth = lips_now > teeth_now
        teeth_above_jaw = teeth_now > jaw_now
        lips_below_teeth = lips_now < teeth_now
        teeth_below_jaw = teeth_now < jaw_now
        
        # Entry conditions
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume + price > trend
            if lips_above_teeth and teeth_above_jaw and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume + price < trend
            elif lips_below_teeth and teeth_below_jaw and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips < Teeth or price < trend
            if lips_now < teeth_now or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Lips > Teeth or price > trend
            if lips_now > teeth_now or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1dTrend"
timeframe = "12h"
leverage = 1.0