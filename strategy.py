#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Camarilla pivot reversal with weekly EMA trend filter and volume confirmation.
# Camarilla levels provide precise support/resistance for reversals; weekly EMA filters for trend alignment.
# Volume confirms institutional interest. Designed for 1-2 trades per month per symbol to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "exp_13264_1d_camarilla_pivot_reversal_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    range_val = high - low
    pivot = (high + low + close) / 3
    r4 = pivot + (range_val * CAMARILLA_MULT * 1.5)
    r3 = pivot + (range_val * CAMARILLA_MULT * 1.25)
    r2 = pivot + (range_val * CAMARILLA_MULT * 1.166)
    r1 = pivot + (range_val * CAMARILLA_MULT * 1.083)
    s1 = pivot - (range_val * CAMARILLA_MULT * 1.083)
    s2 = pivot - (range_val * CAMARILLA_MULT * 1.166)
    s3 = pivot - (range_val * CAMARILLA_MULT * 1.25)
    s4 = pivot - (range_val * CAMARILLA_MULT * 1.5)
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, 50)  # 50-week EMA for strong trend filter
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
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
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Calculate Camarilla levels for today
        r4, r3, r2, r1, pivot, s1, s2, s3, s4 = calculate_camarilla(high[i], low[i], close[i])
        
        # Reversal signals at Camarilla levels
        # Long setup: price touches S3/S4 in uptrend with volume
        long_setup = volume_ok and uptrend and (low[i] <= s3 or low[i] <= s4)
        # Short setup: price touches R3/R4 in downtrend with volume
        short_setup = volume_ok and downtrend and (high[i] >= r3 or high[i] >= r4)
        
        # Generate signals
        if position == 0:
            if long_setup:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_setup:
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