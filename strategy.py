#!/usr/bin/env python3
"""
Experiment #10514: 1h EMA Reversal with 4h/1d Trend Filter and Session Filter
Hypothesis: In trending markets (4h and 1d aligned), short-term reversals at the 1h timeframe provide high-probability entries. 
The strategy uses 4h and 1d EMA alignment to determine trend direction, then looks for pullbacks to the 21 EMA on 1h for entry.
Volume confirmation and session filter (08-20 UTC) reduce noise. Designed to work in both bull and bear markets by following the trend.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10514_1h_ema_reversal_4h_1d_trend_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
VOLUME_SPIKE_MULTIPLIER = 1.3
SIGNAL_SIZE = 0.20
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

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs for trend filter
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    ema_4h_fast = calculate_ema(close_4h, EMA_FAST)
    ema_4h_slow = calculate_ema(close_4h, EMA_SLOW)
    ema_1d_fast = calculate_ema(close_1d, EMA_FAST)
    ema_1d_slow = calculate_ema(close_1d, EMA_SLOW)
    
    # Align to 1h timeframe
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    ema_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_fast)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    ema_1h_fast = calculate_ema(close, EMA_FAST)
    ema_1h_medium = calculate_ema(close, EMA_MEDIUM)
    ema_1h_slow = calculate_ema(close, EMA_SLOW)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, 20) + 1
    
    for i in range(start, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if any EMA not available
        if (np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]) or
            np.isnan(ema_1d_fast_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]) or
            np.isnan(ema_1h_fast[i]) or np.isnan(ema_1h_slow[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: 4h and 1d EMAs aligned
        uptrend_4h = ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i]
        uptrend_1d = ema_1d_fast_aligned[i] > ema_1d_slow_aligned[i]
        downtrend_4h = ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i]
        downtrend_1d = ema_1d_fast_aligned[i] < ema_1d_slow_aligned[i]
        
        # Volume spike confirmation
        volume_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 20 else volume[i]
        volume_spike = volume[i] > (volume_ma * VOLUME_SPIKE_MULTIPLIER)
        
        # 1h EMA alignment for entry timing
        # Long: price pulls back to EMA_MEDIUM in uptrend
        # Short: price pulls back to EMA_MEDIUM in downtrend
        near_ema_medium = abs(close[i] - ema_1h_medium[i]) < (0.5 * atr[i])
        
        # Entry conditions
        long_entry = uptrend_4h and uptrend_1d and volume_spike and near_ema_medium and (close[i] > ema_1h_medium[i])
        short_entry = downtrend_4h and downtrend_1d and volume_spike and near_ema_medium and (close[i] < ema_1h_medium[i])
        
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