#!/usr/bin/env python3
"""
Hypothesis: 6h Elliott Wave-inspired structure using 1d pivot points and Fibonacci ratios.
Long when price retraces to 61.8% Fibonacci level from daily pivot in uptrend (price > 1d EMA50).
Short when price retraces to 38.2% Fibonacci level in downtrend (price < 1d EMA50).
Uses volume confirmation to avoid false signals. Designed for 15-35 trades/year.
"""
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
    
    # Load daily data for pivot and EMA - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard: (H+L+C)/3)
    pivot = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3.0
    # Calculate daily EMA50 for trend filter
    ema50 = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate 6-period high/low for swing measurement (used for Fibonacci)
    high_6 = pd.Series(high).rolling(window=6, min_periods=6).max()
    low_6 = pd.Series(low).rolling(window=6, min_periods=6).min()
    range_6 = high_6 - low_6
    
    # Fibonacci levels from 6-period swing
    fib_382 = low_6 + 0.382 * range_6
    fib_618 = low_6 + 0.618 * range_6
    
    # Align daily levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot.values)
    ema50_aligned = align_htf_to_ltf(prices, df_daily, ema50.values)
    fib_382_aligned = align_htf_to_ltf(prices, df_daily, fib_382.values)
    fib_618_aligned = align_htf_to_ltf(prices, df_daily, fib_618.values)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(6, n):  # Start after 6-period lookback
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price at 61.8% fib level in uptrend (above daily EMA50) with volume
            if (close[i] >= fib_618_aligned[i] * 0.995 and  # Allow small tolerance
                close[i] <= fib_618_aligned[i] * 1.005 and
                close[i] > ema50_aligned[i] and  # Uptrend filter
                volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price at 38.2% fib level in downtrend (below daily EMA50) with volume
            elif (close[i] >= fib_382_aligned[i] * 0.995 and
                  close[i] <= fib_382_aligned[i] * 1.005 and
                  close[i] < ema50_aligned[i] and  # Downtrend filter
                  volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches pivot or breaks below EMA50
                if close[i] >= pivot_aligned[i] or close[i] < ema50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches pivot or breaks above EMA50
                if close[i] <= pivot_aligned[i] or close[i] > ema50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElliottFib_Pivot_EMA50"
timeframe = "6h"
leverage = 1.0
#%%