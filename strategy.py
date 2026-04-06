#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12647_6d_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_PERIOD = 1  # daily
CAMARILLA_MULT = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
CONSECUTIVE_CLOSES = 3  # closes beyond R4/S4 for breakout

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

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    R4 = close + (range_ * CAMARILLA_MULT * 1.5)
    R3 = close + (range_ * CAMARILLA_MULT * 1.25)
    R2 = close + (range_ * CAMARILLA_MULT * 1.1)
    R1 = close + (range_ * CAMARILLA_MULT * 0.5)
    S1 = close - (range_ * CAMARILLA_MULT * 0.5)
    S2 = close - (range_ * CAMARILLA_MULT * 1.1)
    S3 = close - (range_ * CAMARILLA_MULT * 1.25)
    S4 = close - (range_ * CAMARILLA_MULT * 1.5)
    return R4, R3, R2, R1, pivot, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R4_1d, R3_1d, R2_1d, R1_1d, pivot_1d, S1_1d, S2_1d, S3_1d, S4_1d = \
        calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Align daily levels to 6h timeframe
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
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
    consecutive_high = 0
    consecutive_low = 0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily levels not available
        if np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]):
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
        
        # Fade at extreme levels (R3/S3) - mean reversion
        fade_at_R3 = close[i] >= R3_1d_aligned[i]
        fade_at_S3 = close[i] <= S3_1d_aligned[i]
        
        # Breakout confirmation at R4/S4 - need consecutive closes beyond levels
        if close[i] > R4_1d_aligned[i]:
            consecutive_high += 1
            consecutive_low = 0
        elif close[i] < S4_1d_aligned[i]:
            consecutive_low += 1
            consecutive_high = 0
        else:
            consecutive_high = 0
            consecutive_low = 0
        
        breakout_long = consecutive_high >= CONSECUTIVE_CLOSES
        breakout_short = consecutive_low >= CONSECUTIVE_CLOSES
        
        # Entry conditions
        long_entry = volume_ok and fade_at_S3  # fade from S3 support
        short_entry = volume_ok and fade_at_R3  # fade from R3 resistance
        
        # Exit fade positions on breakout
        exit_long_fade = breakout_short
        exit_short_fade = breakout_long
        
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
            if exit_long_fade:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if exit_short_fade:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals