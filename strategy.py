#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Fibonacci_Extension_Retest_1dTrend_Volume"
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
    
    # Get daily data for Fibonacci levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily swing high and low for Fibonacci retracement
    # Using 20-day lookback for swing points
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    
    # Rolling window for swing high/low (20 days)
    roll_high = high_series.rolling(window=20, min_periods=20).max().values
    roll_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate Fibonacci retracement levels (38.2% and 61.8%)
    diff = roll_high - roll_low
    fib_382 = roll_high - diff * 0.382  # 38.2% retracement level
    fib_618 = roll_high - diff * 0.618  # 61.8% retracement level
    
    # Trend filter: 1-day EMA 50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 6h volume > 1.5 * 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align all to 6h timeframe
    fib_382_6h = align_htf_to_ltf(prices, df_1d, fib_382)
    fib_618_6h = align_htf_to_ltf(prices, df_1d, fib_618)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(fib_382_6h[i]) or np.isnan(fib_618_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        fib_382_val = fib_382_6h[i]
        fib_618_val = fib_618_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price retraces to 61.8% Fib level with volume and above trend
            if low[i] <= fib_618_val and close[i] > fib_618_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price retraces to 38.2% Fib level with volume and below trend
            elif high[i] >= fib_382_val and close[i] < fib_382_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches 38.2% Fib level or closes below 61.8%
            if high[i] >= fib_382_val or close[i] < fib_618_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches 61.8% Fib level or closes above 38.2%
            if low[i] <= fib_618_val or close[i] > fib_382_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals