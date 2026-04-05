#!/usr/bin/env python3
"""
Experiment #9734: 1h Donchian Breakout + 4h Trend Filter + Volume Spike + Session Filter
Hypothesis: 1h price breaking 20-period Donchian channel with 4h EMA trend confirmation 
and volume spike during active London/NY session (08-20 UTC) provides high-probability 
trend continuation entries. Uses 4h EMA for trend filter to avoid counter-trend trades, 
reducing whipsaw in choppy markets. Targets 60-150 trades over 4 years (15-37/year) 
by requiring multiple confluence factors. Works in bull markets (breakouts above 4h EMA) 
and bear markets (breakouts below 4h EMA) with session filter avoiding low-liquidity hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9734_1h_donchian_breakout_4h_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
SESSION_START_HOUR = 8   # 08:00 UTC
SESSION_END_HOUR = 20    # 20:00 UTC
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend filter)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h_fast = calculate_ema(close_4h, EMA_FAST)
    ema_4h_slow = calculate_ema(close_4h, EMA_SLOW)
    
    # Align 4h EMA to 1h timeframe (already shifted by 1 in align_htf_to_ltf)
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    
    # Determine trend: bullish if fast > slow, bearish if fast < slow
    trend_bullish = ema_4h_fast_aligned > ema_4h_slow_aligned
    trend_bearish = ema_4h_fast_aligned < ema_4h_slow_aligned
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = pd.to_datetime(open_time).hour
    in_session = (hours >= SESSION_START_HOUR) & (hours <= SESSION_END_HOUR)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, EMA_SLOW) + 1
    
    for i in range(start, n):
        # Skip if not in session
        if not in_session[i]:
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
        
        # Volume spike confirmation
        volume_spike = (not np.isnan(volume_ma[i])) and (volume[i] > volume_ma[i] * VOLUME_SPIKE_MULTIPLIER)
        
        # Skip if trend data not available
        if np.isnan(trend_bullish[i]) or np.isnan(trend_bearish[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Entry conditions
        long_breakout = close[i] >= donch_upper[i]
        short_breakout = close[i] <= donch_lower[i]
        
        # Require trend alignment: long only in bullish trend, short only in bearish trend
        long_entry = long_breakout and volume_spike and trend_bullish[i]
        short_entry = short_breakout and volume_spike and trend_bearish[i]
        
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