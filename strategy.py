#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12799_6d_camarilla_pivot_volume"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    pivot = (high + low + close) / 3
    range_hl = high - low
    r4 = close + range_hl * 1.1 / 2
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    r3_prev, r4_prev, s3_prev, s4_prev = calculate_camarilla(high_1d, low_1d, close_1d)
    r3_prev = np.roll(r3_prev, 1)
    r4_prev = np.roll(r4_prev, 1)
    s3_prev = np.roll(s3_prev, 1)
    s4_prev = np.roll(s4_prev, 1)
    # First day has no previous day
    r3_prev[0] = r4_prev[0] = s3_prev[0] = s4_prev[0] = np.nan
    
    # Align to 6h timeframe
    r3_prev_aligned = align_htf_to_ltf(prices, df_1d, r3_prev)
    r4_prev_aligned = align_htf_to_ltf(prices, df_1d, r4_prev)
    s3_prev_aligned = align_htf_to_ltf(prices, df_1d, s3_prev)
    s4_prev_aligned = align_htf_to_ltf(prices, df_1d, s4_prev)
    
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
        if np.isnan(r3_prev_aligned[i]) or np.isnan(s3_prev_aligned[i]):
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
        
        # Fade at S3/R3, breakout at S4/R4
        fade_long = volume_ok and close[i] <= s3_prev_aligned[i]
        fade_short = volume_ok and close[i] >= r3_prev_aligned[i]
        breakout_long = volume_ok and close[i] >= r4_prev_aligned[i]
        breakout_short = volume_ok and close[i] <= s4_prev_aligned[i]
        
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