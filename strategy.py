#!/usr/bin/env python3
"""
6h_1w_1d_Retracement_Fibonacci
Hypothesis: During weekly uptrends, price retraces to 0.618 Fibonacci level from weekly swing low to high; 
during weekly downtrends, price retraces to 0.382 level from weekly swing high to low. 
Enter in direction of weekly trend at these retracement levels with 1d volume confirmation. 
Uses 6h for precise entry timing. Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in both bull (buy retracements in uptrend) and bear (sell retracements in downtrend) markets.
"""

name = "6h_1w_1d_Retracement_Fibonacci"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend and swing points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- Weekly Trend: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Weekly Swing Points (using previous week's high/low) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Arrays to store swing high/low for each week
    swing_high = np.full_like(close_1w, np.nan)
    swing_low = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        # Use previous week's high/low as swing points
        swing_high[i] = high_1w[i-1]
        swing_low[i] = low_1w[i-1]
    
    # Calculate Fibonacci retracement levels from previous week's swing
    # For uptrend: retrace from swing low to swing high
    # For downtrend: retrace from swing high to swing low
    diff = swing_high - swing_low
    fib_0618 = swing_low + 0.618 * diff  # 61.8% retracement in uptrend
    fib_0382 = swing_high - 0.382 * diff  # 38.2% retracement in downtrend
    
    # Align Fibonacci levels to 6h timeframe
    fib_0618_6h = align_htf_to_ltf(prices, df_1w, fib_0618)
    fib_0382_6h = align_htf_to_ltf(prices, df_1w, fib_0382)
    
    # --- 1d Volume Confirmation: volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(fib_0618_6h[i]) or np.isnan(fib_0382_6h[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close_6h[i] > ema50_1w_aligned[i]
        trend_down = close_6h[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume
            if trend_up and vol_ok:
                # Long: price at 61.8% retracement level during uptrend
                if abs(close_6h[i] - fib_0618_6h[i]) < 0.001 * close_6h[i]:  # within 0.1%
                    signals[i] = 0.25
                    position = 1
            elif trend_down and vol_ok:
                # Short: price at 38.2% retracement level during downtrend
                if abs(close_6h[i] - fib_0382_6h[i]) < 0.001 * close_6h[i]:  # within 0.1%
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: reverse signal or contrary weekly trend
            if position == 1:
                # Exit long: weekly trend turns down or price reaches opposite extreme
                if trend_down or close_6h[i] >= swing_high[-1] if not np.isnan(swing_high[-1]) else False:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up or price reaches opposite extreme
                if trend_up or close_6h[i] <= swing_low[-1] if not np.isnan(swing_low[-1]) else False:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals