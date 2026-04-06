#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Camarilla pivot levels on 1d timeframe with volume confirmation on 6h
# Uses Camarilla's mathematically derived support/resistance levels (R3/S3 for mean reversion, R4/S4 for breakout)
# Works in bull/bear because it captures both mean reversion in ranges and breakouts in trends
# Volume confirmation filters weak signals. Target: 75-150 trades over 4 years (19-38/year).

name = "exp_12911_6h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    pivot = (high + low + close) / 3.0
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return pivot, r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot points
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    pivot_vals = np.zeros(len(close_d))
    r3_vals = np.zeros(len(close_d))
    r4_vals = np.zeros(len(close_d))
    s3_vals = np.zeros(len(close_d))
    s4_vals = np.zeros(len(close_d))
    
    for i in range(len(close_d)):
        pivot, r3, r4, s3, s4 = calculate_camarilla(high_d[i], low_d[i], close_d[i])
        pivot_vals[i] = pivot
        r3_vals[i] = r3
        r4_vals[i] = r4
        s3_vals[i] = s3
        s4_vals[i] = s4
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot_vals)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_vals)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_vals)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_vals)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_vals)
    
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
        # Skip if Camarilla levels not available
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
        
        # Mean reversion at S3/R3 OR breakout at S4/R4
        mean_reversion_long = volume_ok and close[i] <= s3_aligned[i] and close[i] > s4_aligned[i]
        mean_reversion_short = volume_ok and close[i] >= r3_aligned[i] and close[i] < r4_aligned[i]
        breakout_long = volume_ok and close[i] >= r4_aligned[i]
        breakout_short = volume_ok and close[i] <= s4_aligned[i]
        
        # Generate signals
        if position == 0:
            if mean_reversion_long or breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif mean_reversion_short or breakout_short:
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