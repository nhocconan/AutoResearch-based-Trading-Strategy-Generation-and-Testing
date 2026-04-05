#!/usr/bin/env python3
"""
Experiment #9744: 1d Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: Donchian(20) breakouts on the daily timeframe, filtered by weekly EMA trend direction and volume spikes, provide high-probability trend-following entries. Works in bull markets via breakouts above upper band and in bear markets via breakdowns below lower band. Volume confirmation reduces false breakouts. Targets 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9744_1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).values

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
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA trend
    weekly_close = df_weekly['close'].values
    ema_fast_weekly = calculate_ema(weekly_close, EMA_FAST)
    ema_slow_weekly = calculate_ema(weekly_close, EMA_SLOW)
    weekly_trend_up = ema_fast_weekly > ema_slow_weekly
    
    # Align weekly trend to daily timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up.astype(float))
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_trend_up_aligned[i]):
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
        
        # Weekly trend filter
        trend_up = weekly_trend_up_aligned[i] > 0.5
        
        # Breakout conditions
        breakout_up = close[i] > donch_upper[i]
        breakout_down = close[i] < donch_lower[i]
        
        # Entry conditions: breakout in direction of weekly trend + volume spike
        long_entry = breakout_up and trend_up and volume_spike
        short_entry = breakout_down and (not trend_up) and volume_spike
        
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