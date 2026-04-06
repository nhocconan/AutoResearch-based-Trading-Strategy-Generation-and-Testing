#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12531_6d_camarilla1d_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
CLOSE_THRESHOLD = 0.5
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * CAMARILLA_MULTIPLIER / 2
    r3 = close + range_val * CAMARILLA_MULTIPLIER / 4
    r2 = close + range_val * CAMARILLA_MULTIPLIER / 6
    r1 = close + range_val * CAMARILLA_MULTIPLIER / 12
    s1 = close - range_val * CAMARILLA_MULTIPLIER / 12
    s2 = close - range_val * CAMARILLA_MULTIPLIER / 6
    s3 = close - range_val * CAMARILLA_MULTIPLIER / 4
    s4 = close - range_val * CAMARILLA_MULTIPLIER / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
        
        # Camarilla-based entry conditions
        # Long: price closes above R3 with volume (breakout continuation)
        long_breakout = close[i] > r3_aligned[i] and close[i] > r2_aligned[i] and close[i-1] <= r3_aligned[i-1]
        # Short: price closes below S3 with volume (breakdown continuation)
        short_breakout = close[i] < s3_aligned[i] and close[i] < s2_aligned[i] and close[i-1] >= s3_aligned[i-1]
        
        # Fade conditions (mean reversion at extremes)
        # Long fade: price touches S4 and reverses up
        long_fade = close[i] >= s4_aligned[i] and close[i] < s3_aligned[i] and close[i-1] < s4_aligned[i-1]
        # Short fade: price touches R4 and reverses down
        short_fade = close[i] <= r4_aligned[i] and close[i] > r3_aligned[i] and close[i-1] > r4_aligned[i-1]
        
        # Combined entry conditions
        long_entry = volume_ok and (long_breakout or long_fade)
        short_entry = volume_ok and (short_breakout or short_fade)
        
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