#!/usr/bin/env python3
"""
Experiment #9618: 1d Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: Daily Donchian(20) breakouts aligned with weekly trend provide high-probability
trend-following signals with volume confirmation to filter false breakouts. Weekly trend
ensures we trade in the direction of the higher timeframe momentum, reducing whipsaw.
Targets 30-100 total trades over 4 years (7-25/year) by requiring both Donchian breakout
and weekly trend alignment, which should occur infrequently enough to minimize fee drag
while capturing significant trends. Works in bull markets (breakouts above weekly MA)
and bear markets (breakouts below weekly MA) by using weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9618_1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_MA_PERIOD = 50  # Weekly 50-period MA for trend
VOLUME_CONFIRM_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ma(close, period):
    """Calculate simple moving average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly moving average for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ma = calculate_ma(weekly_close, WEEKLY_MA_PERIOD)
    weekly_ma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ma)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_CONFIRM_PERIOD, min_periods=VOLUME_CONFIRM_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_MA_PERIOD, VOLUME_CONFIRM_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly MA not available
        if np.isnan(weekly_ma_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend direction
        weekly_trend_up = close[i] > weekly_ma_aligned[i]
        weekly_trend_down = close[i] < weekly_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # Entry conditions: breakout in direction of weekly trend with volume confirmation
        long_entry = breakout_up and weekly_trend_up and volume_ok
        short_entry = breakout_down and weekly_trend_down and volume_ok
        
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