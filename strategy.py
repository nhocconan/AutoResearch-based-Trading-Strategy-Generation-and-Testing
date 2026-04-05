#!/usr/bin/env python3
"""
Experiment #10534: 1h Trend Pullback with 4h/1d Confirmation
Hypothesis: In trending markets (identified by 4h/1d EMA alignment), pullbacks to the 1h EMA21
provide high-probability entries. Long when price > 4h EMA50 > 1d EMA50 and pulls back to 1h EMA21.
Short when price < 4h EMA50 < 1d EMA50 and pulls back to 1h EMA21. Volume confirmation filters
weak signals. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10534_1h_trend_pullback_4h_1d_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 21      # 1h EMA for entry
EMA_MEDIUM = 50    # 4h EMA for trend
EMA_SLOW = 50      # 1d EMA for higher timeframe trend
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.20
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

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF EMAs
    ema_4h = calculate_ema(df_4h['close'].values, EMA_MEDIUM)
    ema_1d = calculate_ema(df_1d['close'].values, EMA_SLOW)
    
    # Align HTF EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    ema_fast = calculate_ema(close, EMA_FAST)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_FAST, EMA_MEDIUM, EMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF EMA not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend alignment: 4h and 1d EMA in same direction
        bullish_alignment = ema_4h_aligned[i] > ema_1d_aligned[i]
        bearish_alignment = ema_4h_aligned[i] < ema_1d_aligned[i]
        
        # Pullback to 1h EMA21
        pullback_long = close[i] <= ema_fast[i] * 1.005  # within 0.5% above EMA
        pullback_short = close[i] >= ema_fast[i] * 0.995  # within 0.5% below EMA
        
        # Entry conditions
        long_entry = bullish_alignment and pullback_long and volume_spike
        short_entry = bearish_alignment and pullback_short and volume_spike
        
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