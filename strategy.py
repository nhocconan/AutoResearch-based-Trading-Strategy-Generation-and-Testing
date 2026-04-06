#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot support/resistance and volume confirmation
# Works in bull/bear because Donchian breakouts capture strong momentum moves,
# daily pivots provide key institutional levels that hold across regimes,
# volume filters out false breakouts, and ATR stops limit drawdowns.
# Target: 80-150 total trades over 4 years (20-38/year) for statistical significance.

name = "exp_13007_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Calculate Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points"""
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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian channel
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    donchian_upper_d, donchian_lower_d = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    
    # Calculate daily pivot points
    close_d = df_daily['close'].values
    pivot_vals = np.zeros(len(close_d))
    r1_vals = np.zeros(len(close_d))
    r2_vals = np.zeros(len(close_d))
    r3_vals = np.zeros(len(close_d))
    s1_vals = np.zeros(len(close_d))
    s2_vals = np.zeros(len(close_d))
    s3_vals = np.zeros(len(close_d))
    
    for i in range(len(close_d)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_d[i], low_d[i], close_d[i])
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
    
    # Align to 6h timeframe
    donchian_upper_d_aligned = align_htf_to_ltf(prices, df_daily, donchian_upper_d)
    donchian_lower_d_aligned = align_htf_to_ltf(prices, df_daily, donchian_lower_d)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_vals)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_vals)
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily levels not available
        if (np.isnan(donchian_upper_d_aligned[i]) or np.isnan(donchian_lower_d_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
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
        
        # Breakout above daily Donchian upper with volume
        breakout_long = volume_ok and close[i] >= donchian_upper_d_aligned[i]
        # Breakdown below daily Donchian lower with volume
        breakout_short = volume_ok and close[i] <= donchian_lower_d_aligned[i]
        
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