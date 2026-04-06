#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12655_6d_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 10  # Lookback for weekly pivot calculation
PIVOT_NEAREST = 5    # How many pivots to consider for support/resistance
BREAKOUT_BUFFER = 0.001  # 0.1% buffer to avoid whipsaws
VOLUME_MULTIPLIER = 1.5  # Volume must be 1.5x average
VOLUME_LOOKBACK = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points (standard formula)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOLUME_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(pivot_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Find nearest support/resistance levels
        # For longs: look for resistance above, for shorts: look for support below
        current_price = close[i]
        
        # Calculate distances to pivot levels
        dist_to_r1 = (r1_aligned[i] - current_price) / current_price if not np.isnan(r1_aligned[i]) else np.inf
        dist_to_r2 = (r2_aligned[i] - current_price) / current_price if not np.isnan(r2_aligned[i]) else np.inf
        dist_to_r3 = (r3_aligned[i] - current_price) / current_price if not np.isnan(r3_aligned[i]) else np.inf
        dist_to_s1 = (current_price - s1_aligned[i]) / current_price if not np.isnan(s1_aligned[i]) else np.inf
        dist_to_s2 = (current_price - s2_aligned[i]) / current_price if not np.isnan(s2_aligned[i]) else np.inf
        dist_to_s3 = (current_price - s3_aligned[i]) / current_price if not np.isnan(s3_aligned[i]) else np.inf
        
        # Long conditions: break above resistance with volume
        # Consider breakout if price is within buffer of resistance and moving up
        near_r1 = dist_to_r1 > -BREAKOUT_BUFFER and dist_to_r1 < BREAKOUT_BUFFER
        near_r2 = dist_to_r2 > -BREAKOUT_BUFFER and dist_to_r2 < BREAKOUT_BUFFER
        near_r3 = dist_to_r3 > -BREAKOUT_BUFFER and dist_to_r3 < BREAKOUT_BUFFER
        
        # Short conditions: break below support with volume
        near_s1 = dist_to_s1 > -BREAKOUT_BUFFER and dist_to_s1 < BREAKOUT_BUFFER
        near_s2 = dist_to_s2 > -BREAKOUT_BUFFER and dist_to_s2 < BREAKOUT_BUFFER
        near_s3 = dist_to_s3 > -BREAKOUT_BUFFER and dist_to_s3 < BREAKOUT_BUFFER
        
        # Price momentum check (ensure we're actually breaking through)
        price_up = close[i] > close[i-1]
        price_down = close[i] < close[i-1]
        
        long_breakout = volume_ok and (near_r1 or near_r2 or near_r3) and price_up
        short_breakout = volume_ok and (near_s1 or near_s2 or near_s3) and price_down
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout:
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