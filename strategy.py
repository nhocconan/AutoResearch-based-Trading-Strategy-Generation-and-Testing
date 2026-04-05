#!/usr/bin/env python3
"""
Experiment #9574: 1h Donchian Breakout + Volume Spike + Trend Filter.
Hypothesis: 1h timeframe is noisy; use 4h/1d for trend direction and 1h only for entry timing.
Buy when price breaks 4h Donchian high with volume spike and 1d uptrend.
Sell when price breaks 4h Donchian low with volume spike and 1d downtrend.
Targets 60-150 total trades over 4 years (15-38/year) to minimize fee drag.
Works in bull (breakouts) and bear (breakdowns) with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9574_1h_donchian_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
TREND_EMA_FAST = 9
TREND_EMA_SLOW = 21
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load HTF data ONCE before loop (4h for Donchian, 1d for trend)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper, donchian_lower = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_1h = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_1h = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_fast_1d = calculate_ema(close_1d, TREND_EMA_FAST)
    ema_slow_1d = calculate_ema(close_1d, TREND_EMA_SLOW)
    
    # Align 1d EMA trend to 1h timeframe
    ema_fast_1h = align_htf_to_ltf(prices, df_1d, ema_fast_1d)
    ema_slow_1h = align_htf_to_ltf(prices, df_1d, ema_slow_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_SLOW, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or
            np.isnan(ema_fast_1h[i]) or np.isnan(ema_slow_1h[i])):
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
        
        # Trend filter: 1d EMA fast > slow = uptrend, < = downtrend
        uptrend = ema_fast_1h[i] > ema_slow_1h[i]
        downtrend = ema_fast_1h[i] < ema_slow_1h[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper_1h[i]
        breakout_down = close[i] < donchian_lower_1h[i]
        
        # Entry conditions
        long_entry = uptrend and volume_spike and breakout_up
        short_entry = downtrend and volume_spike and breakout_down
        
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