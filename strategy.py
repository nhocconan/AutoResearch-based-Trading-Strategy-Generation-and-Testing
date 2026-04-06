#!/usr/bin/env python3
"""
Experiment #12299: 6h Camarilla Pivot + Volume + Trend Filter
Hypothesis: Use daily Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
combined with 12h EMA trend filter and volume confirmation. Works in bull/bear by
switching between mean reversion at extreme levels and breakout continuation.
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12299_6h_camarilla_pivot_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close, lookback):
    """Calculate Camarilla pivot levels for each bar"""
    # Use rolling window of lookback period
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]  # handle first bar
    
    # Pivot point (standard)
    pivot = (high_max + low_min + close_prev) / 3.0
    range_val = high_max - low_min
    
    # Camarilla levels
    r4 = close_prev + (range_val * 1.1 / 2)
    r3 = close_prev + (range_val * 1.1 / 4)
    s3 = close_prev - (range_val * 1.1 / 4)
    s4 = close_prev - (range_val * 1.1 / 2)
    
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    ema_12h = calculate_ema(df_12h['close'].values, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    r4, r3, s3, s4 = calculate_camarilla(high, low, close, CAMARILLA_LOOKBACK)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_LOOKBACK, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Trend filter (12h)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Camarilla conditions
        # Mean reversion at S3/R3 (extreme levels)
        long_mean_revert = close[i] <= s3[i] and close[i] > s4[i]  # between S3 and S4
        short_mean_revert = close[i] >= r3[i] and close[i] < r4[i]  # between R3 and R4
        
        # Breakout continuation at S4/R4
        long_breakout = close[i] > s4[i]  # break above S4
        short_breakout = close[i] < r4[i]  # break below R4
        
        # Entry conditions: mean reversion in ranging, breakout in trending
        # In uptrend: look for long mean reversion at S3 or breakout above S4
        # In downtrend: look for short mean reversion at R3 or breakdown below R4
        long_entry = volume_ok and uptrend_12h and (long_mean_revert or long_breakout)
        short_entry = volume_ok and downtrend_12h and (short_mean_revert or short_breakout)
        
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
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals