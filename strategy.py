#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) to detect trend direction and strength
# Long when Lips > Teeth > Jaw (bullish alignment) with price above Lips and volume confirmation
# Short when Lips < Teeth < Jaw (bearish alignment) with price below Lips and volume confirmation
# Filters trades using 1d EMA50 trend direction to avoid counter-trend whipsaws
# Targets 15-25 trades per year for low fee drag (< 100 total over 4 years)
# Alligator excels in trending markets while avoiding sideways chop, ideal for BTC/ETH

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (13,8,5 smoothed with 8,5,3 periods)
    # Jaw (Blue): 13-period SMMA smoothed by 8 periods
    # Teeth (Red): 8-period SMMA smoothed by 5 periods  
    # Lips (Green): 5-period SMMA smoothed by 3 periods
    
    def smma(arr, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw_raw = smma(high, 13)  # Using high for jaw as per Williams
    teeth_raw = smma(high, 8)  # Using high for teeth
    lips_raw = smma(high, 5)   # Using high for lips
    
    jaw = smma(jaw_raw, 8)    # Smooth jaw by 8
    teeth = smma(teeth_raw, 5) # Smooth teeth by 5
    lips = smma(lips_raw, 3)   # Smooth lips by 3
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = ema50_1d[1:] - ema50_1d[:-1]  # slope: positive = uptrend
    ema50_1d_slope = np.concatenate([[0], ema50_1d_slope])  # align length
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50_val = ema50_1d_aligned[i]
        ema50_slope = ema50_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: bullish alignment (Lips > Teeth > Jaw) + price above Lips + volume + 1d uptrend
            if (lips_val > teeth_val > jaw_val and 
                close_val > lips_val and 
                vol_conf_val and 
                ema50_slope > 0):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (Lips < Teeth < Jaw) + price below Lips + volume + 1d downtrend
            elif (lips_val < teeth_val < jaw_val and 
                  close_val < lips_val and 
                  vol_conf_val and 
                  ema50_slope < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or price crosses below Teeth or 1d trend turns down
            if (lips_val < teeth_val or 
                close_val < teeth_val or 
                ema50_slope < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment or price crosses above Teeth or 1d trend turns up
            if (lips_val > teeth_val or 
                close_val > teeth_val or 
                ema50_slope > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals