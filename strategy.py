#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
# In both bull and bear markets, price tends to revert from R3/S3 levels and break through R4/S4.
# Fade at R3/S3 (mean reversion) with volume confirmation, breakout at R4/S4 (trend continuation).
# Uses 1d Camarilla levels for institutional reference points. Target: 75-150 total trades over 4 years.

name = "exp_13091_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # Use previous day's OHLC
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r4, r3, r2, r1, s1, s2, s3, s4

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    r4_1d, r3_1d, r2_1d, r1_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Previous day's levels (shifted by 1)
    r4_1d_prev = np.roll(r4_1d, 1)
    r3_1d_prev = np.roll(r3_1d, 1)
    r2_1d_prev = np.roll(r2_1d, 1)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    s2_1d_prev = np.roll(s2_1d, 1)
    s3_1d_prev = np.roll(s3_1d, 1)
    s4_1d_prev = np.roll(s4_1d, 1)
    
    # Set first day's values to NaN (no previous day)
    r4_1d_prev[0] = np.nan
    r3_1d_prev[0] = np.nan
    r2_1d_prev[0] = np.nan
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    s2_1d_prev[0] = np.nan
    s3_1d_prev[0] = np.nan
    s4_1d_prev[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d_prev)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_prev)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d_prev)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d_prev)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_prev)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d_prev)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]):
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
        
        # Fade at R3/S3 (mean reversion)
        fade_short = volume_ok and (close[i] >= r3_1d_aligned[i]) and (i == 0 or close[i-1] < r3_1d_aligned[i-1])
        fade_long = volume_ok and (close[i] <= s3_1d_aligned[i]) and (i == 0 or close[i-1] > s3_1d_aligned[i-1])
        
        # Breakout at R4/S4 (trend continuation)
        breakout_long = volume_ok and (close[i] >= r4_1d_aligned[i]) and (i == 0 or close[i-1] < r4_1d_aligned[i-1])
        breakout_short = volume_ok and (close[i] <= s4_1d_aligned[i]) and (i == 0 or close[i-1] > s4_1d_aligned[i-1])
        
        # Generate signals
        if position == 0:
            if fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
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