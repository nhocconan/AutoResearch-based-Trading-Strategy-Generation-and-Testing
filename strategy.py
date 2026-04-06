#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12784_1d_1w_donchian_volume"
timeframe = "1d"
leverage = 1.0

# Parameters - optimized for 1d timeframe
WEEKLY_DONCHIAN_PERIOD = 10  # 10-week Donchian for weekly trend
DAILY_DONCHIAN_PERIOD = 20   # 20-day Donchian for entry
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels for trend filter
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly upper/lower bands (10-week lookback)
    weekly_upper = pd.Series(weekly_high).rolling(window=WEEKLY_DONCHIAN_PERIOD, min_periods=WEEKLY_DONCHIAN_PERIOD).max().values
    weekly_lower = pd.Series(weekly_low).rolling(window=WEEKLY_DONCHIAN_PERIOD, min_periods=WEEKLY_DONCHIAN_PERIOD).min().values
    
    # Align weekly bands to daily timeframe
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    
    # Load daily data for entry signals
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian channels for entry
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Daily upper/lower bands (20-day lookback)
    daily_upper = pd.Series(daily_high).rolling(window=DAILY_DONCHIAN_PERIOD, min_periods=DAILY_DONCHIAN_PERIOD).max().values
    daily_lower = pd.Series(daily_low).rolling(window=DAILY_DONCHIAN_PERIOD, min_periods=DAILY_DONCHIAN_PERIOD).min().values
    
    # Align daily bands to daily timeframe (identity but keeps consistency)
    daily_upper_aligned = align_htf_to_ltf(prices, df_daily, daily_upper)
    daily_lower_aligned = align_htf_to_ltf(prices, df_daily, daily_lower)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_DONCHIAN_PERIOD, DAILY_DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if bands not available
        if (np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or
            np.isnan(daily_upper_aligned[i]) or np.isnan(daily_lower_aligned[i])):
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
        
        # Trend filter: price must be above/below weekly Donchian for long/short
        weekly_trend_up = close[i] > weekly_upper_aligned[i]
        weekly_trend_down = close[i] < weekly_lower_aligned[i]
        
        # Breakout signals with trend filter and volume
        breakout_long = volume_ok and weekly_trend_up and close[i] >= daily_upper_aligned[i]
        breakout_short = volume_ok and weekly_trend_down and close[i] <= daily_lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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