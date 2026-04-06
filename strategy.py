#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h/1d filters and session control
# Uses 4h EMA for trend direction, 1d Donchian breakout for entry timing,
# and 08-20 UTC session filter to avoid low-liquidity hours.
# Designed for 60-150 trades over 4 years (15-38/year) to minimize fee drag.
# Works in bull/bear because trend filter adapts to direction and breakouts
# capture strong moves in either regime.

name = "exp_12934_1h_trend_breakout_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 9
EMA_SLOW = 21
DONCHIAN_PERIOD = 20
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, span):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=span, adjust=False, min_periods=span).mean().values

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_fast_4h = calculate_ema(close_4h, EMA_FAST)
    ema_slow_4h = calculate_ema(close_4h, EMA_SLOW)
    trend_4h = ema_fast_4h - ema_slow_4h  # Positive = uptrend, Negative = downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d Donchian channels for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper_1d, donchian_lower_1d = calculate_donchian(high_1d, low_1d, DONCHIAN_PERIOD)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, DONCHIAN_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]):
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
        
        # Trend and breakout conditions
        uptrend = trend_4h_aligned[i] > 0
        downtrend = trend_4h_aligned[i] < 0
        breakout_up = close[i] > donchian_upper_aligned[i]
        breakout_down = close[i] < donchian_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if uptrend and breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif downtrend and breakout_down:
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