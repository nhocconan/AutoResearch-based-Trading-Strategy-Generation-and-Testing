#!/usr/bin/env python3
"""
Experiment #9504: 1d Donchian Breakout + Volume Spike + Regime Filter.
Hypothesis: Donchian(20) breakouts on 1d timeframe, filtered by volume spikes and 1w trend (via 1w EMA), provide high-probability entries in both bull and bear markets. 
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag while capturing trend moves.
Works in bull (breakouts above 20-day high) and bear (breakdowns below 20-day low).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9504_1d_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
SIGNAL_SIZE = 0.30
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian(high, low, period):
    """Calculate Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter (fast and slow)
    close_1w = df_1w['close'].values
    ema_fast_1w = calculate_ema(close_1w, EMA_FAST_PERIOD)
    ema_slow_1w = calculate_ema(close_1w, EMA_SLOW_PERIOD)
    
    # Align 1w EMA to 1d timeframe
    ema_fast_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_fast_1w)
    ema_slow_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slow_1w)
    
    # Calculate 1d Donchian channel
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    start = max(DONCHIAN_PERIOD, EMA_FAST_PERIOD, EMA_SLOW_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_fast_1w_aligned[i]) or np.isnan(ema_slow_1w_aligned[i]):
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
        
        # Trend filter: EMA fast > EMA slow = uptrend, EMA fast < EMA slow = downtrend
        uptrend = ema_fast_1w_aligned[i] > ema_slow_1w_aligned[i]
        downtrend = ema_fast_1w_aligned[i] < ema_slow_1w_aligned[i]
        
        # Entry conditions
        long_entry = volume_spike and close[i] >= donchian_upper[i] and uptrend
        short_entry = volume_spike and close[i] <= donchian_lower[i] and downtrend
        
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
</response>