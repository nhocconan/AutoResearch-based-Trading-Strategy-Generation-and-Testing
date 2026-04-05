#!/usr/bin/env python3
"""
Experiment #9615: 6h Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: Donchian(20) breakouts on 6h timeframe, filtered by weekly trend direction (from 1w close vs 4-week ago close) 
and confirmed by volume spikes, capture medium-term trends while avoiding whipsaws. 
Weekly trend filter ensures we only take longs in uptrends and shorts in downtrends, 
working in both bull (breakout continuation) and bear (breakdown continuation) markets.
Targets 100-200 total trades over 4 years (25-50/year) for optimal balance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_9615_6h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
WEEKLY_TREND_LOOKBACK = 4  # 4 weeks ago
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly trend: close > close 4 weeks ago = uptrend
    weekly_close = df_1w['close'].values
    weekly_trend_up = weekly_close >= np.roll(weekly_close, WEEKLY_TREND_LOOKBACK)
    weekly_trend_down = weekly_close <= np.roll(weekly_close, WEEKLY_TREND_LOOKBACK)
    
    # Align weekly trend to 6h timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + WEEKLY_TREND_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]):
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
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper[i]
        bearish_breakout = close[i] < donchian_lower[i]
        
        # Entry conditions with weekly trend filter
        long_entry = bullish_breakout and weekly_trend_up_aligned[i] > 0.5 and volume_spike
        short_entry = bearish_breakout and weekly_trend_down_aligned[i] > 0.5 and volume_spike
        
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