#!/usr/bin/env python3
"""
Experiment #11771: 6h Camarilla Pivot Fade with Volume Confirmation
Hypothesis: Price often reverts from Camarilla R3/S3 levels during ranging markets, while
breaking R4/S4 indicates strong continuation. Volume confirms institutional participation.
Works in ranging markets (fade) and trending markets (breakout). Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11771_6h_camarilla_pivot_fade_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
PIVOT_BUFFER = 0.001  # 0.1% buffer to avoid whipsaws

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    # Based on previous day's OHLC
    pivot = (high + low + close) / 3.0
    range_ = high - low
    
    # Resistance levels
    r4 = close + range_ * 1.500
    r3 = close + range_ * 1.250
    r2 = close + range_ * 1.166
    r1 = close + range_ * 1.083
    
    # Support levels
    s1 = close - range_ * 1.083
    s2 = close - range_ * 1.166
    s3 = close - range_ * 1.250
    s4 = close - range_ * 1.500
    
    return r1, r2, r3, r4, s1, s2, s3, s4, pivot

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # First TR is just high-low (no previous close)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels from previous day's OHLC
    # We need to shift by 1 to use previous day's levels for current day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla for each day using previous day's OHLC
    r1_list, r2_list, r3_list, r4_list = [], [], [], []
    s1_list, s2_list, s3_list, s4_list = [], [], [], []
    pivot_list = []
    
    for i in range(len(high_1d)):
        if i == 0:
            # For first day, use same day's OHLC (no previous)
            r1, r2, r3, r4, s1, s2, s3, s4, pivot = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        else:
            # Use previous day's OHLC
            r1, r2, r3, r4, s1, s2, s3, s4, pivot = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r1_list.append(r1)
        r2_list.append(r2)
        r3_list.append(r3)
        r4_list.append(r4)
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)
        s4_list.append(s4)
        pivot_list.append(pivot)
    
    r1_1d = np.array(r1_list)
    r2_1d = np.array(r2_list)
    r3_1d = np.array(r3_list)
    r4_1d = np.array(r4_list)
    s1_1d = np.array(s1_list)
    s2_1d = np.array(s2_list)
    s3_1d = np.array(s3_list)
    s4_1d = np.array(s4_list)
    pivot_1d = np.array(pivot_list)
    
    # Align Camarilla levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
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
        # Skip if 1d data not available
        if np.isnan(pivot_6h[i]):
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
        
        # Price levels with buffer
        r3_level = r3_6h[i] * (1 + PIVOT_BUFFER)
        r4_level = r4_6h[i] * (1 + PIVOT_BUFFER)
        s3_level = s3_6h[i] * (1 - PIVOT_BUFFER)
        s4_level = s4_6h[i] * (1 - PIVOT_BUFFER)
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Fade at R3/S3, breakout at R4/S4
        fade_short = close[i] > r3_level and close[i] < r4_level and volume_ok
        fade_long = close[i] < s3_level and close[i] > s4_level and volume_ok
        breakout_long = close[i] > r4_level and volume_ok
        breakout_short = close[i] < s4_level and volume_ok
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
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