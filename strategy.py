#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week pivot point reversal levels with volume confirmation.
# Buy at S1/S2 support with volume spike, sell at R1/R2 resistance with volume spike.
# Uses weekly pivot for key institutional levels, works in ranging markets (common in 2025).
# Target: 60-120 total trades over 4 years (15-30/year) to avoid excessive fees.

name = "exp_13755_6h_weekly_pivot_reversal_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous week's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate pivot points for each week
    pivot_vals = np.full_like(close_w, np.nan)
    r1_vals = np.full_like(close_w, np.nan)
    r2_vals = np.full_like(close_w, np.nan)
    s1_vals = np.full_like(close_w, np.nan)
    s2_vals = np.full_like(close_w, np.nan)
    
    for i in range(len(close_w)):
        p, r1, r2, s1, s2 = calculate_pivot_points(high_w[i], low_w[i], close_w[i])
        pivot_vals[i] = p
        r1_vals[i] = r1
        r2_vals[i] = r2
        s1_vals[i] = s1
        s2_vals[i] = s2
    
    # Align to 6h timeframe (using previous week's pivot - no lookahead)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2_vals)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2_vals)
    
    # 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i])):
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
        
        # Reversal signals at pivot levels
        # Long at S1/S2 support with volume
        long_at_s1 = volume_ok and close[i] <= s1_aligned[i] * 1.005 and close[i] >= s1_aligned[i] * 0.995
        long_at_s2 = volume_ok and close[i] <= s2_aligned[i] * 1.005 and close[i] >= s2_aligned[i] * 0.995
        
        # Short at R1/R2 resistance with volume
        short_at_r1 = volume_ok and close[i] >= r1_aligned[i] * 0.995 and close[i] <= r1_aligned[i] * 1.005
        short_at_r2 = volume_ok and close[i] >= r2_aligned[i] * 0.995 and close[i] <= r2_aligned[i] * 1.005
        
        long_signal = long_at_s1 or long_at_s2
        short_signal = short_at_r1 or short_at_r2
        
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
            # Exit long at R1 resistance or stop
            if close[i] >= r1_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short at S1 support or stop
            if close[i] <= s1_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals