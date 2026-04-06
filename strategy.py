#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12775_6d_weekly_pivot_breakout"
timeframe = "6h"
leverage = 1.0

# Parameters
WEEKLY_PIVOT_PERIOD = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.5
SIGNAL_SIZE = 0.30
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
BREAKOUT_BUFFER = 0.001  # 0.1% buffer above/below pivot

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot points for each week
    pivot_vals = np.full(len(weekly_high), np.nan)
    r1_vals = np.full(len(weekly_high), np.nan)
    r2_vals = np.full(len(weekly_high), np.nan)
    r3_vals = np.full(len(weekly_high), np.nan)
    s1_vals = np.full(len(weekly_high), np.nan)
    s2_vals = np.full(len(weekly_high), np.nan)
    s3_vals = np.full(len(weekly_high), np.nan)
    
    for i in range(len(weekly_high)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(
            weekly_high[i], weekly_low[i], weekly_close[i]
        )
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_vals)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_vals)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout above R3 or below S3 with volume
        breakout_long = volume_ok and close[i] > (r3_aligned[i] * (1 + BREAKOUT_BUFFER))
        breakout_short = volume_ok and close[i] < (s3_aligned[i] * (1 - BREAKOUT_BUFFER))
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals