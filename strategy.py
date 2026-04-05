#!/usr/bin/env python3
"""
exp_7518_1d_1w_donchian20_volume_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high + weekly MA50 rising + volume > 1.5x average.
Short when price breaks below 20-day low + weekly MA50 falling + volume > 1.5x average.
Targets 40-80 trades over 4 years (10-20/year) with strict breakout conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7518_1d_1w_donchian20_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
WEEKLY_MA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly MA50 for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ma50 = pd.Series(close_weekly).ewm(span=WEEKLY_MA_PERIOD, adjust=False, min_periods=WEEKLY_MA_PERIOD).mean().values
    weekly_ma50_prev = np.roll(weekly_ma50, 1)
    weekly_ma50_prev[0] = weekly_ma50[0]
    weekly_rising = weekly_ma50 > weekly_ma50_prev
    weekly_falling = weekly_ma50 < weekly_ma50_prev
    weekly_rising_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rising)
    weekly_falling_aligned = align_htf_to_ltf(prices, df_weekly, weekly_falling)
    
    # Calculate daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-day high/low)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(weekly_rising_aligned[i]) or np.isnan(weekly_falling_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation
        volume_confirm = volume[i] > VOLUME_MULTIPLIER * avg_volume[i]
        breakout_up = close[i] > highest_high[i-1]  # break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # break below previous period's low
        
        # Entry conditions
        long_entry = breakout_up and weekly_rising_aligned[i] and volume_confirm
        short_entry = breakout_down and weekly_falling_aligned[i] and volume_confirm
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE  # maintain position until stoploss
        elif position == -1:
            signals[i] = -SIGNAL_SIZE  # maintain position until stoploss
    
    return signals