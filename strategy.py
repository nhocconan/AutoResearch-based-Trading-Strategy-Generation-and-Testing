#!/usr/bin/env python3
"""
Experiment #10087: 6h Camarilla Pivot + Volume Spike + ATR Filter
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Long at S3 with volume spike and ATR confirmation; short at R3 with volume spike.
Works in ranging markets (mean reversion at S3/R3) and trending markets (breakouts at S4/R4).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10087_6h_camarilla_pivot_volume_atr"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # daily
VOLUME_SPIKE_MULTIPLIER = 2.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Camarilla formulas
    pivot = (high + low + close) / 3
    range_val = high - low
    
    r4 = close + range_val * 1.500
    r3 = close + range_val * 1.250
    r2 = close + range_val * 1.166
    r1 = close + range_val * 1.083
    
    s1 = close - range_val * 1.083
    s2 = close - range_val * 1.166
    s3 = close - range_val * 1.250
    s4 = close - range_val * 1.500
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    r4_d, r3_d, r2_d, r1_d, p_d, s1_d, s2_d, s3_d, s4_d = calculate_camarilla(daily_high, daily_low, daily_close)
    
    # Align Camarilla levels to 6h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_daily, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_daily, s3_d)
    r4_d_aligned = align_htf_to_ltf(prices, df_daily, r4_d)
    s4_d_aligned = align_htf_to_ltf(prices, df_daily, s4_d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # ATR-based entry filter: only trade when volatility is sufficient
        atr_ratio = atr[i] / np.mean(atr[max(0, i-20):i+1]) if i >= 20 and np.mean(atr[max(0, i-20):i+1]) > 0 else 1.0
        volatility_filter = atr_ratio > 0.8  # Avoid low volatility periods
        
        # Entry conditions
        long_entry = (close[i] <= s3_d_aligned[i]) and volume_spike and volatility_filter
        short_entry = (close[i] >= r3_d_aligned[i]) and volume_spike and volatility_filter
        
        # Breakout entries at S4/R4 for trend continuation
        long_breakout = (close[i] >= s4_d_aligned[i]) and volume_spike and volatility_filter
        short_breakout = (close[i] <= r4_d_aligned[i]) and volume_spike and volatility_filter
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_breakout:
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