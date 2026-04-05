#!/usr/bin/env python3
"""
Experiment #10394: 1h Donchian Breakout with 4h/1d Trend and Volume Spike
Hypothesis: In choppy 1h markets, use 4h and 1d trends as filters to capture
only high-probability breakouts. Volume spikes confirm institutional interest.
Designed for 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
Works in bull markets (breakouts above 4h/1d EMAs) and bear markets 
(breakdowns below 4h/1d EMAs). Session filter (08-20 UTC) reduces noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10394_1h_donchian_breakout_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters - tuned for ~15-37 trades/year
DONCHIAN_PERIOD = 18          # Slightly shorter for more sensitivity on 1h
VOLUME_SPIKE_MULTIPLIER = 2.0 # Higher threshold to reduce false signals
FAST_EMA_PERIOD = 9           # 4h trend
SLOW_EMA_PERIOD = 21          # 1d trend (acts as filter on 4h)
SIGNAL_SIZE = 0.20            # Conservative size to manage drawdown
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs for trend filters
    close_4h = df_4h['close'].values
    ema_4h_fast = calculate_ema(close_4h, FAST_EMA_PERIOD)
    ema_4h_slow = calculate_ema(close_4h, SLOW_EMA_PERIOD)
    
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, SLOW_EMA_PERIOD)  # Same period as 4h slow for consistency
    
    # Align HTF indicators to 1h timeframe (with shift(1) inside align_htf_to_ltf)
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels on 1h
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, FAST_EMA_PERIOD, SLOW_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            # Force flat outside session
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
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
        
        # Determine 4h trend (fast vs slow EMA)
        trend_4h_up = ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i]
        trend_4h_down = ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i]
        
        # Determine 1d trend (price vs EMA)
        trend_1d_up = close[i] > ema_1d_aligned[i]
        trend_1d_down = close[i] < ema_1d_aligned[i]
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: 
        # - 4h and 1d trends must align (both up or both down)
        # - Breakout in direction of aligned trend
        # - Volume spike for confirmation
        long_entry = (trend_4h_up and trend_1d_up and bullish_breakout and volume_spike)
        short_entry = (trend_4h_down and trend_1d_down and bearish_breakout and volume_spike)
        
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