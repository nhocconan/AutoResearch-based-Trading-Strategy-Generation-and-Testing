#!/usr/bin/env python3

"""
12h Williams Alligator with 1d trend filter and volume confirmation.
Goes long when price is above Alligator's Jaw (bullish alignment) with 1d EMA50 uptrend and volume spike.
Goes short when price is below Alligator's Jaw (bearish alignment) with 1d EMA50 downtrend and volume spike.
Exits when price crosses back below/above Jaw. Williams Alligator identifies trend phases,
avoiding whipsaws in ranging markets. Target: 15-30 trades/year per symbol.
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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (blue): 13-period SMMA, smoothed by 8 periods
    # Teeth (red): 8-period SMMA, smoothed by 5 periods  
    # Lips (green): 5-period SMMA, smoothed by 3 periods
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period smoothed
    jaw = smma(jaw, 8)     # further smoothed by 8
    teeth = smma(close, 8)  # 8-period smoothed
    teeth = smma(teeth, 5)  # further smoothed by 5
    lips = smma(close, 5)   # 5-period smoothed
    lips = smma(lips, 3)    # further smoothed by 3
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above Jaw, teeth above jaw, lips above teeth (bullish alignment)
            # Plus daily uptrend and volume spike
            if (close[i] > jaw[i] and 
                teeth[i] > jaw[i] and 
                lips[i] > teeth[i] and
                close[i] > ema50_aligned[i] and  # Daily uptrend
                volume[i] > 2.0 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price below Jaw, teeth below jaw, lips below teeth (bearish alignment)
            # Plus daily downtrend and volume spike
            elif (close[i] < jaw[i] and 
                  teeth[i] < jaw[i] and 
                  lips[i] < teeth[i] and
                  close[i] < ema50_aligned[i] and  # Daily downtrend
                  volume[i] > 2.0 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses Jaw
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Jaw
                if close[i] <= jaw[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Jaw
                if close[i] >= jaw[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0