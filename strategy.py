#!/usr/bin/env python3
"""
Experiment #10674: 1h Donchian Breakout + 4h Trend + Daily Volume Spike
Hypothesis: 1h Donchian breakouts aligned with 4h trend direction, filtered by daily volume spikes,
provide high-probability trend-following entries. Uses 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
Works in bull markets (breakouts above 4h EMA) and bear markets (breakdowns below 4h EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10674_1h_donchian_breakout_4h_trend_daily_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 16
VOLUME_SPIKE_MULTIPLIER = 1.8
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
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
    
    # Load 4h and daily data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction (fast/slow crossover)
    close_4h = df_4h['close'].values
    ema_fast_4h = calculate_ema(close_4h, EMA_FAST_PERIOD)
    ema_slow_4h = calculate_ema(close_4h, EMA_SLOW_PERIOD)
    
    # Align 4h EMAs to 1h timeframe
    ema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_fast_4h)
    ema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slow_4h)
    
    # Calculate daily volume average for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_SLOW_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if 4h EMA not available
        if np.isnan(ema_fast_4h_aligned[i]) or np.isnan(ema_slow_4h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        
        # Volume spike detection (daily volume)
        volume_spike = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Trend filter: 4h EMA crossover
        bullish_trend = ema_fast_4h_aligned[i] > ema_slow_4h_aligned[i]
        bearish_trend = ema_fast_4h_aligned[i] < ema_slow_4h_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 4h trend with daily volume spike
        long_entry = bullish_breakout and bullish_trend and volume_spike
        short_entry = bearish_breakout and bearish_trend and volume_spike
        
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