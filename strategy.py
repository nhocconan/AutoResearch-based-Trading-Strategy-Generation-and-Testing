#!/usr/bin/env python3
"""
1d Williams Alligator with 1-week Trend Filter and Volume Confirmation.
Long when price > Alligator teeth + weekly trend up + volume spike.
Short when price < Alligator teeth + weekly trend down + volume spike.
Exit when price crosses Alligator teeth or weekly trend changes.
Williams Alligator uses smoothed moving averages (SMMA) of median price.
Designed for low frequency (7-25 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result.fill(np.nan)
    if n < period:
        return result
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema_50_1w = np.empty_like(weekly_close, dtype=np.float64)
    ema_50_1w.fill(np.nan)
    if len(weekly_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(weekly_close[:50])
        for i in range(50, len(weekly_close)):
            ema_50_1w[i] = alpha * weekly_close[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: three SMMA lines of median price
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward  
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply forward shift (to avoid look-ahead)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    for i in range(8, len(jaw)):
        jaw[i] = jaw_raw[i-8]  # shifted 8 bars forward
    for i in range(5, len(teeth)):
        teeth[i] = teeth_raw[i-5]  # shifted 5 bars forward
    for i in range(3, len(lips)):
        lips[i] = lips_raw[i-3]  # shifted 3 bars forward
    
    # Align weekly trend to lower timeframe
    # (already aligned in the EMA calculation above)
    
    # Volume filter: volume > 1.5x average (calculated from 1d volume MA20)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13+8=21 periods) and weekly EMA (50 periods)
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(teeth[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        teeth_val = teeth[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price above teeth + weekly trend up + volume
            if price_now > teeth_val and price_now > weekly_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price below teeth + weekly trend down + volume
            elif price_now < teeth_val and price_now < weekly_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth or weekly trend turns down
            if price_now < teeth_val or price_now < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above teeth or weekly trend turns up
            if price_now > teeth_val or price_now > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0