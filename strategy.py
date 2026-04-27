#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with daily volume confirmation and 12h trend filter.
Enters long when price is above Alligator's teeth (green line) with volume above average and 12h uptrend.
Enters short when price is below Alligator's teeth with volume above average and 12h downtrend.
Alligator uses smoothed moving averages (SMMA) of median price (HL/2) with specific periods.
Williams Alligator is designed to identify trends and filter out ranging markets, making it suitable
for both bull and bear markets when combined with higher timeframe trend filter.
Target: 20-40 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if length <= 0:
        return np.full_like(source, np.nan)
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (prev SMMA * (length-1) + current price) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate median price (HL/2) for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator lines (all SMMA)
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    jaw = smma(median_price, 13)
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    teeth = smma(median_price, 8)
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    lips = smma(median_price, 5)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h close for trend filter
    close_12h = df_12h['close'].values
    # Use close price directly for trend (no need for MA to avoid lag)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator components (lips needs 5-period SMMA + 3 shift = 8 min)
    # Jaw: 13+8=21, Teeth: 8+5=13, Lips: 5+3=8 -> max is 21
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_12h = close_12h_aligned[i]
        
        # Current Alligator values
        lips_now = lips[i]
        teeth_now = teeth[i]
        jaw_now = jaw[i]
        
        # Volume filter: volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: price vs teeth (green line) with volume + 12h trend
        if position == 0:
            # Long: price above teeth with volume + 12h uptrend
            if price_now > teeth_now and vol_filter and price_now > trend_12h:
                signals[i] = size
                position = 1
            # Short: price below teeth with volume + 12h downtrend
            elif price_now < teeth_now and vol_filter and price_now < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth or 12h trend turns down
            if price_now < teeth_now or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above teeth or 12h trend turns up
            if price_now > teeth_now or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_1dVolume_12hTrend"
timeframe = "4h"
leverage = 1.0