#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13967_6d_1d_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h breakout in direction of 1d pivot levels (R4/S4) with volume confirmation.
# Uses 1d Camarilla pivot levels: R4/S4 as strong breakout levels, R3/S3 as fade levels.
# Long when price breaks above R4 with volume > 1.5x average, short when breaks below S4.
# Exit when price returns to R3/S3 or opposite pivot level is touched.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (breaks above R4 with volume) and bear (breaks below S4 with volume).

def calculate_pivots(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r4 = close + range_ * 1.1 / 2
    r3 = close + range_ * 1.1 / 4
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    return r4, r3, s3, s4

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
    
    # Load 1d data for pivot calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_pivots(high_1d, low_1d, close_1d)
    
    # Align pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data for breakout detection, ATR, and volume
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
    start = max(50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Breakout signals from pivot levels
        breakout_r4 = close[i] > r4_aligned[i-1]  # break above R4
        breakdown_s4 = close[i] < s4_aligned[i-1]  # break below S4
        
        # Mean reversion signals (fade at R3/S3)
        fade_r3 = close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1]  # cross below R3
        fade_s3 = close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1]  # cross above S3
        
        # Entry signals
        long_signal = breakout_r4 and volume_ok
        short_signal = breakdown_s4 and volume_ok
        
        # Exit signals (mean reversion or opposite breakout)
        exit_long = fade_r3 or breakdown_s4
        exit_short = fade_s3 or breakout_r4
        
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
            # Exit long on mean reversion or opposite breakout
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on mean reversion or opposite breakout
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals