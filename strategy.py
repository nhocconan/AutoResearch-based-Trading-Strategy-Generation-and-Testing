#!/usr/bin/env python3
"""
exp_13895_6d_weekly_pivot_vol_v1
Hypothesis: 6h strategy using weekly pivot levels (from 1w data) for directional bias,
with volume confirmation and ATR-based stops. Weekly pivots provide institutional
support/resistance levels that work in both bull and bear markets by identifying
key turning points. Volume confirmation ensures momentum behind moves.
Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13895_6h_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous week's data for pivot calculation
VOLUME_MA = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
PIVOT_BUFFER = 0.001  # 0.1% buffer to avoid whipsaws

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, R3, S1, S2, S3"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

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
    
    # Calculate weekly pivot points from previous week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for each week
    pivots = np.zeros(len(weekly_high))
    r1 = np.zeros(len(weekly_high))
    r2 = np.zeros(len(weekly_high))
    r3 = np.zeros(len(weekly_high))
    s1 = np.zeros(len(weekly_high))
    s2 = np.zeros(len(weekly_high))
    s3 = np.zeros(len(weekly_high))
    
    for i in range(len(weekly_high)):
        p, r1_val, r2_val, r3_val, s1_val, s2_val, s3_val = calculate_pivot_points(
            weekly_high[i], weekly_low[i], weekly_close[i]
        )
        pivots[i] = p
        r1[i] = r1_val
        r2[i] = r2_val
        r3[i] = r3_val
        s1[i] = s1_val
        s2[i] = s2_val
        s3[i] = s3_val
    
    # Align weekly pivot data to 6h timeframe (shifted by 1 week for no look-ahead)
    pivots_aligned = align_htf_to_ltf(prices, df_weekly, pivots)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h data for price, volume, and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivots_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Price levels with buffer to avoid whipsaws
        pivot_level = pivots_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        # Entry logic:
        # Long when price crosses above R1 with volume (breakout)
        # Short when price crosses below S1 with volume (breakdown)
        long_signal = volume_ok and close[i] > (r1_level * (1 + PIVOT_BUFFER)) and close[i-1] <= (r1_level * (1 + PIVOT_BUFFER))
        short_signal = volume_ok and close[i] < (s1_level * (1 - PIVOT_BUFFER)) and close[i-1] >= (s1_level * (1 - PIVOT_BUFFER))
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price returns to pivot level (mean reversion) or stop hit
            if close[i] <= pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price returns to pivot level (mean reversion) or stop hit
            if close[i] >= pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals