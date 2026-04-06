#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Uses daily Camarilla levels (H4/L4) for mean reversion entries when price reaches extreme levels.
# Trend filter from 1d EMA ensures we trade with the higher timeframe momentum.
# Volume confirmation confirms institutional participation at key levels.
# Works in both bull and bear markets by fading extremes in ranging conditions and
# following breakouts in trending markets. Target: 75-150 total trades over 4 years.

name = "exp_13451_6h_camarilla_rev_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Camarilla equations
    pivot = (high + low + close) / 3
    range_val = high - low
    
    # Resistance levels
    r4 = close + range_val * CAMARILLA_MULT * 1.1 / 2
    r3 = close + range_val * CAMARILLA_MULT * 1.1 / 4
    r2 = close + range_val * CAMARILLA_MULT * 1.1 / 6
    r1 = close + range_val * CAMARILLA_MULT * 1.1 / 12
    
    # Support levels
    s1 = close - range_val * CAMARILLA_MULT * 1.1 / 12
    s2 = close - range_val * CAMARILLA_MULT * 1.1 / 6
    s3 = close - range_val * CAMARILLA_MULT * 1.1 / 4
    s4 = close - range_val * CAMARILLA_MULT * 1.1 * 1.1 / 2
    
    return {
        'r4': r4, 'r3': r3, 'r2': r2, 'r1': r1,
        's1': s1, 's2': s2, 's3': s3, 's4': s4,
        'pivot': pivot
    }

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_data = calculate_camarilla(high_1d, low_1d, close_1d_arr)
    
    # Extract Camarilla levels as arrays
    r4 = camarilla_data['r4']
    r3 = camarilla_data['r3']
    s3 = camarilla_data['s3']
    s4 = camarilla_data['s4']
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Trend filter: price above/below 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Camarilla reversal signals
        # Long when price touches S3/S4 and shows rejection
        long_signal = volume_ok and (
            (low[i] <= s3_aligned[i] and close[i] > open[i]) or  # Bullish rejection at S3
            (low[i] <= s4_aligned[i] and close[i] > open[i])   # Bullish rejection at S4
        )
        
        # Short when price touches R3/R4 and shows rejection
        short_signal = volume_ok and (
            (high[i] >= r3_aligned[i] and close[i] < open[i]) or  # Bearish rejection at R3
            (high[i] >= r4_aligned[i] and close[i] < open[i])   # Bearish rejection at R4
        )
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals