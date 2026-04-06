#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13911_6h_camarilla1d_pivot_fade_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOLUME_MA = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    tp = (high + low + close) / 3.0
    # Range
    r = high - low
    # Camarilla levels
    r3 = tp + (r * 1.1 / 4)
    r4 = tp + (r * 1.1 / 2)
    s3 = tp - (r * 1.1 / 4)
    s4 = tp - (r * 1.1 / 2)
    return r3, r4, s3, s4

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
    
    # Load 1d data for Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r3_1d, r4_1d, s3_1d, s4_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
        if np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or np.isnan(volume_ma[i]):
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
        
        # Camarilla levels
        r3 = r3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Entry signals: fade at S3/R3
        long_signal = volume_ok and close[i] <= s3
        short_signal = volume_ok and close[i] >= r3
        
        # Exit signals: take profit at opposite R3/S3
        exit_long = volume_ok and close[i] >= r3
        exit_short = volume_ok and close[i] <= s3
        
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
            # Exit long at R3 or stop loss
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short at S3 or stop loss
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals