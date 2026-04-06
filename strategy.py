#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13955_6d_weekly_pivot_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h price action reacting to weekly pivot levels.
# Fade at weekly R3/S3 (mean reversion), breakout continuation at R4/S4.
# Uses weekly pivot points calculated from prior week's OHLC.
# Entry conditions:
#   - Long: price crosses above S3 with rejection (close > S3 and low touched S3) OR break above R4
#   - Short: price crosses below R3 with rejection (close < R3 and high touched R3) OR break below S4
# Volume confirmation: current volume > 1.5x 20-period average.
# Stop loss: 2x ATR(14) from entry.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets via mean reversion at extremes and breakout continuation.

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # 6h data for price action and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(30, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Price action signals
        touched_s3 = low[i] <= s3_aligned[i]  # touched or went below S3
        touched_r3 = high[i] >= r3_aligned[i]  # touched or went above R3
        close_above_s3 = close[i] > s3_aligned[i]
        close_below_r3 = close[i] < r3_aligned[i]
        break_above_r4 = close[i] > r4_aligned[i]
        break_below_s4 = close[i] < s4_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Entry signals
        # Long: bounce from S3 (rejection) OR break above R4
        long_rejection = touched_s3 and close_above_s3 and volume_ok
        long_breakout = break_above_r4 and volume_ok
        long_signal = long_rejection or long_breakout
        
        # Short: rejection at R3 OR break below S4
        short_rejection = touched_r3 and close_below_r3 and volume_ok
        short_breakout = break_below_s4 and volume_ok
        short_signal = short_rejection or short_breakout
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown below S3 or stop loss
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on breakout above R3 or stop loss
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals