#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 38.2% Fibonacci retracement pullback strategy with 1d EMA200 trend filter and volume confirmation.
# In strong trends (bull or bear), price often pulls back to 38.2% Fibonacci level before continuing.
# Uses 1d swing high/low to calculate Fibonacci levels, aligned to 4h.
# Entry: price touches 38.2% level in trend direction with volume confirmation.
# Exit: price crosses 61.8% Fibonacci level or 1d EMA200.
# Designed for low turnover in both bull and bear markets by targeting high-probability pullbacks.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for swing points and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d swing high and low (50-period lookback for significant swings)
    # Swing high: highest high in last 50 periods
    swing_high_1d = pd.Series(high_1d).rolling(window=50, min_periods=50).max().values
    # Swing low: lowest low in last 50 periods
    swing_low_1d = pd.Series(low_1d).rolling(window=50, min_periods=50).min().values
    
    # Calculate Fibonacci levels: 38.2% and 61.8% retracement
    # In uptrend: retracement from swing high to swing low
    # In downtrend: retracement from swing low to swing high
    diff_1d = swing_high_1d - swing_low_1d
    fib_382_1d = swing_high_1d - 0.382 * diff_1d  # 38.2% level
    fib_618_1d = swing_high_1d - 0.618 * diff_1d  # 61.8% level
    
    # Align 1d indicators to 4h
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    fib_382_4h = align_htf_to_ltf(prices, df_1d, fib_382_1d)
    fib_618_4h = align_htf_to_ltf(prices, df_1d, fib_618_1d)
    swing_high_4h = align_htf_to_ltf(prices, df_1d, swing_high_1d)
    swing_low_4h = align_htf_to_ltf(prices, df_1d, swing_low_1d)
    
    # Volume filter: current volume > 1.8 * 20-period average (moderate filter)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 50-period swing calculations + EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_4h[i]) or 
            np.isnan(fib_382_4h[i]) or 
            np.isnan(fib_618_4h[i]) or 
            np.isnan(swing_high_4h[i]) or 
            np.isnan(swing_low_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Determine trend based on price vs EMA200
        price_above_ema200 = close[i] > ema200_4h[i]
        price_below_ema200 = close[i] < ema200_4h[i]
        
        # Price relative to Fibonacci levels
        price_near_382 = np.abs(close[i] - fib_382_4h[i]) < (0.005 * close[i])  # Within 0.5% of 38.2% level
        price_above_618 = close[i] > fib_618_4h[i]
        price_below_618 = close[i] < fib_618_4h[i]
        
        if position == 0:
            # Long: In uptrend (price > EMA200), pullback to 38.2% level with volume
            if (price_above_ema200 and price_near_382 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: In downtrend (price < EMA200), pullback to 38.2% level with volume
            elif (price_below_ema200 and price_near_382 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below 61.8% level OR crosses below EMA200
            if (price_below_618) or (close[i] < ema200_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above 61.8% level OR crosses above EMA200
            if (price_above_618) or (close[i] > ema200_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Fib382_Pullback_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0