#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily volume confirmation and ATR stoploss.
# Uses 1d as primary timeframe and 1w for trend filter. Breakouts capture strong moves,
# volume filters weak signals, and weekly trend ensures we trade with higher timeframe momentum.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and cost.

name = "exp_13058_1d_weekly_donchian_vol"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_DONCHIAN_PERIOD = 20
DAILY_VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_highest = pd.Series(weekly_high).rolling(window=WEEKLY_DONCHIAN_PERIOD, min_periods=WEEKLY_DONCHIAN_PERIOD).max().values
    weekly_lowest = pd.Series(weekly_low).rolling(window=WEEKLY_DONCHIAN_PERIOD, min_periods=WEEKLY_DONCHIAN_PERIOD).min().values
    
    # Align weekly Donchian to daily
    weekly_highest_aligned = align_htf_to_ltf(prices, df_1w, weekly_highest)
    weekly_lowest_aligned = align_htf_to_ltf(prices, df_1w, weekly_lowest)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily volume MA
    volume_ma = pd.Series(volume).rolling(window=DAILY_VOLUME_MA_PERIOD, min_periods=DAILY_VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_DONCHIAN_PERIOD, DAILY_VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly Donchian not available
        if np.isnan(weekly_highest_aligned[i]) or np.isnan(weekly_lowest_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout signals
        breakout_up = volume_ok and (high[i] > weekly_highest_aligned[i])
        breakout_down = volume_ok and (low[i] < weekly_lowest_aligned[i])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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