#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian(20) breakout with daily EMA(50) filter and volume confirmation.
# Captures strong weekly trends aligned with daily momentum. Volume ensures institutional participation.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 30-100 total trades over 4 years (7-25/year).

name = "exp_13570_1d_weekly_donchian20_daily_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_DONCHIAN_PERIOD = 20
DAILY_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_highest_high = pd.Series(high_1w).rolling(window=WEEKLY_DONCHIAN_PERIOD, min_periods=WEEKLY_DONCHIAN_PERIOD).max().values
    weekly_lowest_low = pd.Series(low_1w).rolling(window=WEEKLY_DONCHIAN_PERIOD, min_periods=WEEKLY_DONCHIAN_PERIOD).min().values
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend filter
    daily_ema = calculate_ema(close, DAILY_EMA_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_DONCHIAN_PERIOD, DAILY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly Donchian or daily EMA not available
        if np.isnan(weekly_highest_high[i]) or np.isnan(weekly_lowest_low[i]) or np.isnan(daily_ema[i]):
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
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > daily_ema[i]
        downtrend = close[i] < daily_ema[i]
        
        # Breakout signals using weekly Donchian channels
        breakout_up = volume_ok and uptrend and (high[i] > weekly_highest_high[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < weekly_lowest_low[i-1])
        
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