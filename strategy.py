# Hypothesis: 4h timeframe strategy using daily Fibonacci retracement levels from the previous week's high-low range, combined with EMA trend filter and volume confirmation. 
# In both bull and bear markets, price tends to retrace to key Fibonacci levels (0.382, 0.618) before continuing the trend. 
# Weekly high-low provides relevant range for the current market regime. 
# Entry: Long when price crosses above 0.618 Fibonacci level with above-average volume and price > daily EMA50. 
# Short when price crosses below 0.382 Fibonacci level with above-average volume and price < daily EMA50. 
# Exit: Price crosses back below/above the 0.5 Fibonacci level or crosses below/above daily EMA50.
# Using weekly lookback ensures the Fibonacci levels are based on completed weekly ranges, avoiding look-ahead.
# Position size: 0.25 to limit drawdown and reduce trade frequency.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high and low (5 trading days)
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    
    # Calculate Fibonacci levels from weekly range
    weekly_range = high_5d - low_5d
    fib_0_382 = low_5d + 0.382 * weekly_range
    fib_0_5 = low_5d + 0.5 * weekly_range
    fib_0_618 = low_5d + 0.618 * weekly_range
    
    # Align Fibonacci levels to 4h timeframe
    fib_0_382_4h = align_htf_to_ltf(prices, df_1d, fib_0_382)
    fib_0_5_4h = align_htf_to_ltf(prices, df_1d, fib_0_5)
    fib_0_618_4h = align_htf_to_ltf(prices, df_1d, fib_0_618)
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 80  # Need weekly high/low, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(fib_0_382_4h[i]) or 
            np.isnan(fib_0_5_4h[i]) or 
            np.isnan(fib_0_618_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Price relative to Fibonacci levels
        price_above_618 = close[i] > fib_0_618_4h[i]
        price_below_382 = close[i] < fib_0_382_4h[i]
        price_below_5 = close[i] < fib_0_5_4h[i]
        price_above_5 = close[i] > fib_0_5_4h[i]
        
        # Trend filter: price relative to daily EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        if position == 0:
            # Long: Price breaks above 0.618 Fibonacci with volume and above EMA50
            if (price_above_618 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 0.382 Fibonacci with volume and below EMA50
            elif (price_below_382 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 0.5 Fibonacci OR below EMA50
            if (price_below_5) or (price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 0.5 Fibonacci OR above EMA50
            if (price_above_5) or (price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Fibonacci_Retracement_EMA50_Volume"
timeframe = "4h"
leverage = 1.0