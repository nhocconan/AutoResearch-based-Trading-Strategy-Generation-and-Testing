#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and trend filter
# Works in bull/bear because breakouts capture strong moves, volume filters weak signals,
# and 1d trend filter ensures alignment with higher timeframe momentum.
# Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag.

name = "exp_12985_12h_donchian20_1d_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
TREND_EMA_FAST = 9
TREND_EMA_SLOW = 21
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

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily indicators
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    volume_d = df_daily['volume'].values
    
    # Daily EMA trend
    ema_fast_d = calculate_ema(close_d, TREND_EMA_FAST)
    ema_slow_d = calculate_ema(close_d, TREND_EMA_SLOW)
    trend_up_d = ema_fast_d > ema_slow_d
    trend_down_d = ema_fast_d < ema_slow_d
    
    # Align daily indicators to 12h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_daily, trend_up_d)
    trend_down_aligned = align_htf_to_ltf(prices, df_daily, trend_down_d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels
    donch_up, donch_low = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # 12h volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # 12h ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, TREND_EMA_SLOW) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(donch_up[i]) or np.isnan(donch_low[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Trend alignment from daily
        trend_up = trend_up_aligned[i] if not np.isnan(trend_up_aligned[i]) else False
        trend_down = trend_down_aligned[i] if not np.isnan(trend_down_aligned[i]) else False
        
        # Breakout conditions
        breakout_up = volume_ok and close[i] >= donch_up[i]
        breakout_down = volume_ok and close[i] <= donch_low[i]
        
        # Generate signals
        if position == 0:
            if breakout_up and trend_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down and trend_down:
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