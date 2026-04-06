#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day pivot direction and volume confirmation.
# Uses daily pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) to filter breakouts.
# In bull markets, breakouts above R4 with volume capture strong uptrends.
# In bear markets, breakdowns below S4 with volume capture strong downtrends.
# Daily pivot direction ensures alignment with higher timeframe structure.
# Volume filters out false breakouts. Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13267_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, etc."""
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
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points for previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points for each day
    pivot_vals = np.full_like(high_1d, np.nan)
    r3_vals = np.full_like(high_1d, np.nan)
    s3_vals = np.full_like(high_1d, np.nan)
    r4_vals = np.full_like(high_1d, np.nan)
    s4_vals = np.full_like(high_1d, np.nan)
    
    for i in range(len(high_1d)):
        pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
        pivot_vals[i] = pivot
        r3_vals[i] = r3
        s3_vals[i] = s3
        r4_vals[i] = r4
        s4_vals[i] = s4
    
    # Align to 6h timeframe (using previous day's pivot to avoid look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_vals)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Pivot-based conditions
        # Use previous day's levels (already aligned with shift(1) in align_htf_to_ltf)
        pivot = pivot_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # Determine pivot bias: bullish if price > pivot, bearish if price < pivot
        bullish_bias = close[i] > pivot
        bearish_bias = close[i] < pivot
        
        # Breakout signals with pivot filters
        breakout_up = volume_ok and bullish_bias and (high[i] > highest_high[i-1]) and (close[i] > r4)
        breakout_down = volume_ok and bearish_bias and (low[i] < lowest_low[i-1]) and (close[i] < s4)
        
        # Mean reversion signals at S3/R3 (optional, for range markets)
        # Not used in this version to keep signals infrequent
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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