#!/usr/bin/env python3
# 6h_1d_fibonacci_reversal_zone
# Hypothesis: Price reversals at 61.8% and 38.2% Fibonacci retracement levels of the daily range, confirmed by volume and momentum divergence.
# Works in both bull and bear markets by fading extreme moves at key retracement levels with confirmation filters.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.

name = "6h_1d_fibonacci_reversal_zone"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Fibonacci levels and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range and Fibonacci levels (based on previous day's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    daily_range = prev_high - prev_low
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    
    # Fibonacci retracement levels (38.2% and 61.8%)
    fib_382 = prev_close + 0.382 * daily_range  # 38.2% retracement from close
    fib_618 = prev_close + 0.618 * daily_range  # 61.8% retracement from close
    
    # For downtrends, we need inverse levels
    fib_382_inv = prev_close - 0.382 * daily_range
    fib_618_inv = prev_close - 0.618 * daily_range
    
    # Momentum confirmation: RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Align Fibonacci levels and RSI to 6h timeframe
    fib_382_a = align_htf_to_ltf(prices, df_1d, fib_382)
    fib_618_a = align_htf_to_ltf(prices, df_1d, fib_618)
    fib_382_inv_a = align_htf_to_ltf(prices, df_1d, fib_382_inv)
    fib_618_inv_a = align_htf_to_ltf(prices, df_1d, fib_618_inv)
    rsi_a = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(fib_382_a[i]) or np.isnan(fib_618_a[i]) or 
            np.isnan(fib_382_inv_a[i]) or np.isnan(fib_618_inv_a[i]) or
            np.isnan(rsi_a[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price at 61.8% retracement level in uptrend (RSI > 50) or 38.2% in downtrend
        # Short setup: price at 61.8% retracement level in downtrend (RSI < 50) or 38.2% in uptrend
        
        # Long conditions: bounce from support
        long_condition = (
            (close[i] <= fib_618_a[i] * 1.005 and close[i] >= fib_618_a[i] * 0.995 and rsi_a[i] > 50) or  # 61.8% support in uptrend
            (close[i] <= fib_382_inv_a[i] * 1.005 and close[i] >= fib_382_inv_a[i] * 0.995 and rsi_a[i] < 50)  # 38.2% resistance in downtrend (inverted)
        ) and vol_confirm[i]
        
        # Short conditions: rejection from resistance
        short_condition = (
            (close[i] >= fib_618_a[i] * 0.995 and close[i] <= fib_618_a[i] * 1.005 and rsi_a[i] < 50) or  # 61.8% resistance in downtrend
            (close[i] >= fib_382_inv_a[i] * 0.995 and close[i] <= fib_382_inv_a[i] * 1.005 and rsi_a[i] > 50)  # 38.2% support in uptrend (inverted)
        ) and vol_confirm[i]
        
        # Entry logic
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite signal or price moves significantly away from level
        elif position == 1 and (
            close[i] < fib_618_a[i] * 0.98 or  # Broke below 61.8% level
            close[i] > fib_382_a[i] * 1.02     # Went above 38.2% level (taking profits)
        ):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (
            close[i] > fib_618_a[i] * 1.02 or  # Broke above 61.8% level
            close[i] < fib_382_a[i] * 0.98     # Went below 38.2% level (taking profits)
        ):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals