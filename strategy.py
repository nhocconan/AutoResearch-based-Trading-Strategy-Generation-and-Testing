#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Fibonacci_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (only close needed)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Fibonacci retracement calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Identify weekly swing high and low over last 4 weeks
    swing_high = np.full(len(df_1w), np.nan)
    swing_low = np.full(len(df_1w), np.nan)
    
    for i in range(4, len(df_1w)):
        # Look back 4 weeks for highest high and lowest low
        window_high = df_1w['high'].iloc[i-4:i+1].max()
        window_low = df_1w['low'].iloc[i-4:i+1].min()
        swing_high[i] = window_high
        swing_low[i] = window_low
    
    # Calculate Fibonacci retracement levels (38.2% and 61.8%)
    diff = swing_high - swing_low
    fib_382 = swing_high - diff * 0.382  # Resistance in uptrend, support in downtrend
    fib_618 = swing_high - diff * 0.618  # Resistance in uptrend, support in downtrend
    
    # Align all to 6h timeframe
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    fib_382_6h = align_htf_to_ltf(prices, df_1w, fib_382)
    fib_618_6h = align_htf_to_ltf(prices, df_1w, fib_618)
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20, 4)  # Need enough data for EMA50, volume MA, and swing calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_6h[i]) or np.isnan(fib_382_6h[i]) or 
            np.isnan(fib_618_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1w_6h[i]
        fib_382_val = fib_382_6h[i]
        fib_618_val = fib_618_6h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price above weekly EMA50 and breaks above 61.8% fib level with volume
            if close[i] > trend and close[i] > fib_618_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly EMA50 and breaks below 38.2% fib level with volume
            elif close[i] < trend and close[i] < fib_382_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below 38.2% fib level (mean reversion)
            if close[i] < fib_382_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above 61.8% fib level (mean reversion)
            if close[i] > fib_618_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals