#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversals with 1d trend filter and volume confirmation.
# Camarilla levels from prior day: fade at R3/S3 (mean reversion), breakout at R4/S4 (trend continuation).
# Trend filter ensures we trade with higher timeframe momentum. Volume filters weak signals.
# Works in bull/bear: mean reversion in range, trend continuation in breakout.
# Target: 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.

name = "camarilla_6h_1d_pivot_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
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
    """Calculate Camarilla pivot levels for the period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r4 = close + CAMARILLA_MULTIPLIER * range_val * 1.1 / 2
    r3 = close + CAMARILLA_MULTIPLIER * range_val * 1.1 / 4
    s3 = close - CAMARILLA_MULTIPLIER * range_val * 1.1 / 4
    s4 = close - CAMARILLA_MULTIPLIER * range_val * 1.1 / 2
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d, r4_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe (prior day's levels for current day)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Trend filter: EMA50 on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = 50  # EMA period
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Volume confirmation: volume > 1.5x 20-period MA
        volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Camarilla signals
        fade_long = volume_ok and downtrend and (close[i] <= s3_1d_aligned[i]) and (i == 0 or close[i-1] > s3_1d_aligned[i-1])
        fade_short = volume_ok and uptrend and (close[i] >= r3_1d_aligned[i]) and (i == 0 or close[i-1] < r3_1d_aligned[i-1])
        breakout_long = volume_ok and uptrend and (close[i] >= r4_1d_aligned[i]) and (i == 0 or close[i-1] < r4_1d_aligned[i-1])
        breakout_short = volume_ok and downtrend and (close[i] <= s4_1d_aligned[i]) and (i == 0 or close[i-1] > s4_1d_aligned[i-1])
        
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