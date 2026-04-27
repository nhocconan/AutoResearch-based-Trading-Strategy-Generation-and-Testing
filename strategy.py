#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-day trend filter and volume confirmation.
In bull market (price > 1-day EMA50): long when Alligator lines are bullish aligned (jaw < teeth < lips) and volume > 1.5x average.
In bear market (price < 1-day EMA50): short when Alligator lines are bearish aligned (jaw > teeth > lips) and volume > 1.5x average.
Alligator identifies trend, daily trend filters direction, volume confirms participation.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator: three smoothed moving averages (SMMA)"""
    if len(high) < jaw_period + 8 or len(low) < jaw_period + 8:
        return np.full_like(high, np.nan, dtype=np.float64), \
               np.full_like(high, np.nan, dtype=np.float64), \
               np.full_like(high, np.nan, dtype=np.float64)
    
    # Median price
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMMA) - similar to EMA but with different smoothing
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        result = np.full_like(data, np.nan, dtype=np.float64)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    jaw = smma(median_price, jaw_period)   # Blue line (13-period)
    teeth = smma(median_price, teeth_period)  # Red line (8-period)
    lips = smma(median_price, lips_period)    # Green line (5-period)
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend
    daily_close = df_1d['close'].values
    ema_50_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_50_1d.fill(np.nan)
    if len(daily_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(daily_close[:50])
        for i in range(50, len(daily_close)):
            ema_50_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get daily data for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.empty_like(vol_1d, dtype=np.float64)
    vol_ma_20_1d.fill(np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12-hour Alligator
    jaw, teeth, lips = calculate_alligator(high, low, 13, 8, 5)
    jaw_aligned = jaw  # Same timeframe, no alignment needed
    teeth_aligned = teeth
    lips_aligned = lips
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13+8=21 bars), daily EMA50 (50), daily volume MA20 (20)
    start_idx = max(21, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current indicators
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Daily close price for trend comparison
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        if np.isnan(daily_close_aligned[i]):
            signals[i] = 0.0
            continue
        daily_close_val = daily_close_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Alligator alignment conditions
        bullish_aligned = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        bearish_aligned = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Bull market (price > daily EMA50): look for long
            if daily_close_val > ema_trend and bullish_aligned and vol_filter:
                signals[i] = size
                position = 1
            # Bear market (price < daily EMA50): look for short
            elif daily_close_val < ema_trend and bearish_aligned and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or trend change to bear
            if bearish_aligned or daily_close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish alignment or trend change to bull
            if bullish_aligned or daily_close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0