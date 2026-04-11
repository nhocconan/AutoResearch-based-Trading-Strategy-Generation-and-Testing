#!/usr/bin/env python3
"""
6h_1d_fib_retracement_volume_trend_v1
Strategy: 6h Fibonacci retracement with volume confirmation and 1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses 6h Fibonacci retracement levels (38.2%, 61.8%) from the previous 1d swing for entry, with volume confirmation (>1.5x average volume) and filtered by 1d EMA50 trend alignment. Designed to capture pullbacks in trending markets while avoiding false signals in chop. Uses 1d for trend direction and 6h only for timing. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_fib_retracement_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d swing high/low for Fibonacci levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Fibonacci retracement levels from previous day's swing
    # For uptrend: retracement from low to high
    # For downtrend: retracement from high to low
    swing_range = high_1d - low_1d
    fib_382 = low_1d + 0.382 * swing_range
    fib_618 = low_1d + 0.618 * swing_range
    
    # Align Fibonacci levels to 6h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Retracement conditions: price at Fibonacci levels with tolerance
        # Tolerance: 0.5% of price
        tolerance = price_close * 0.005
        at_fib_382 = abs(price_close - fib_382_aligned[i]) <= tolerance
        at_fib_618 = abs(price_close - fib_618_aligned[i]) <= tolerance
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: price at 38.2% or 61.8% retracement in uptrend with volume
        long_signal = (at_fib_382 or at_fib_618) and vol_confirmed and uptrend_1d
        
        # Short: price at 38.2% or 61.8% retracement in downtrend with volume
        short_signal = (at_fib_382 or at_fib_618) and vol_confirmed and downtrend_1d
        
        # Exit when price moves 1% away from Fibonacci level or opposite signal
        exit_long = position == 1 and (
            price_close < fib_382_aligned[i] - tolerance or 
            price_close > fib_618_aligned[i] + tolerance
        )
        exit_short = position == -1 and (
            price_close > fib_382_aligned[i] + tolerance or 
            price_close < fib_618_aligned[i] - tolerance
        )
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals