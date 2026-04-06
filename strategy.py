#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour timeframe with 1-day ATR-based volatility filter and 3-day EMA trend filter.
# Uses volatility contraction/expansion as signal: enter when volatility expands after contraction
# in direction of 3-day EMA trend. Works in both bull/bear markets as it captures momentum bursts
# regardless of direction. Low-frequency (12h) minimizes fee drag while capturing significant moves.

name = "exp_13225_12h_volatility_ema_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
VOLATILITY_LOOKBACK = 10
VOLATILITY_THRESHOLD = 1.5  # Volatility expansion threshold
EMA_FAST = 3
EMA_SLOW = 10
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_1d_ma = pd.Series(atr_1d).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # Calculate 12-hour indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA for trend filter
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ATR_PERIOD + VOLATILITY_LOOKBACK, EMA_SLOW) + 1
    
    for i in range(start, n):
        # Skip if 1D ATR MA not available
        if np.isnan(atr_1d_aligned[i]):
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
        
        # Volatility expansion condition: current ATR > threshold * moving average
        volatility_expansion = atr_1d_aligned[i] > (atr_1d_aligned[i] * VOLATILITY_THRESHOLD) if not np.isnan(atr_1d_aligned[i]) else False
        # Actually: current volatility should be greater than historical average
        vol_ratio = atr_1d[i] / atr_1d_ma[i] if not np.isnan(atr_1d_ma[i]) and atr_1d_ma[i] > 0 else 0
        volatility_expansion = vol_ratio > VOLATILITY_THRESHOLD
        
        # Trend direction
        uptrend = ema_fast[i] > ema_slow[i]
        downtrend = ema_fast[i] < ema_slow[i]
        
        # Entry signals
        if position == 0:
            if volatility_expansion and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            elif volatility_expansion and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals