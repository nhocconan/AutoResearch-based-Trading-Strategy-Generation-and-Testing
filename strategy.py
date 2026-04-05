#!/usr/bin/env python3
"""
Experiment #9773: 4h Donchian(20) Breakout + Volume Confirmation + 12h Trend Filter
Hypothesis: Donchian breakouts with volume confirmation and higher timeframe trend filter
provide high-probability trend-following entries. Targets 75-200 total trades over 4 years
(19-50/year) to balance opportunity and cost. Works in bull (breakouts up) and bear
(breakouts down) with 12h trend filter preventing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9773_4h_donchian20_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
TREND_EMA_FAST = 9
TREND_EMA_SLOW = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load HTF data ONCE before loop (12h for trend filter)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA trend
    close_12h = df_12h['close'].values
    ema_fast_12h = calculate_ema(close_12h, TREND_EMA_FAST)
    ema_slow_12h = calculate_ema(close_12h, TREND_EMA_SLOW)
    
    # Align 12h EMA trend to 4h timeframe
    ema_fast_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_fast_12h)
    ema_slow_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slow_12h)
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20, TREND_EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_fast_12h_aligned[i]) or np.isnan(ema_slow_12h_aligned[i]):
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
        
        # 12h trend filter: fast EMA above slow EMA = uptrend, below = downtrend
        uptrend_12h = ema_fast_12h_aligned[i] > ema_slow_12h_aligned[i]
        downtrend_12h = ema_fast_12h_aligned[i] < ema_slow_12h_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # Entry conditions: breakout + volume + trend alignment
        long_entry = breakout_up and volume_spike and uptrend_12h
        short_entry = breakout_down and volume_spike and downtrend_12h
        
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