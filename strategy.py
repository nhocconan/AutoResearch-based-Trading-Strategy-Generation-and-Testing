#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 12h trend filter on 12h timeframe
# Uses daily breakouts for strong directional moves, filtered by 12h EMA trend to avoid false breakouts
# Volume confirmation ensures breakouts have conviction
# Works in bull/bear because breakouts capture momentum, trend filter avoids whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "exp_12892_12h_donchian20_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
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

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Daily Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    
    # Daily EMA trend
    ema_fast = calculate_ema(close_d, EMA_FAST)
    ema_slow = calculate_ema(close_d, EMA_SLOW)
    
    # Daily volume MA
    volume_ma = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Daily ATR for stop loss
    atr_d = calculate_atr(high_d, low_d, close_d, ATR_PERIOD)
    
    # Align to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_daily, donchian_lower)
    ema_fast_12h = align_htf_to_ltf(prices, df_daily, ema_fast)
    ema_slow_12h = align_htf_to_ltf(prices, df_daily, ema_slow)
    volume_ma_12h = align_htf_to_ltf(prices, df_daily, volume_ma)
    atr_12h = align_htf_to_ltf(prices, df_daily, atr_d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or \
           np.isnan(ema_fast_12h[i]) or np.isnan(ema_slow_12h[i]) or \
           np.isnan(volume_ma_12h[i]) or np.isnan(atr_12h[i]):
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
        
        # Volume confirmation (using 12h volume)
        volume_ok = volume[i] > (volume_ma_12h[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_12h[i]) else False
        
        # Trend filter (using 12h EMA)
        uptrend = ema_fast_12h[i] > ema_slow_12h[i]
        downtrend = ema_fast_12h[i] < ema_slow_12h[i]
        
        # Breakout conditions
        breakout_long = volume_ok and uptrend and close[i] >= donchian_upper_12h[i]
        breakout_short = volume_ok and downtrend and close[i] <= donchian_lower_12h[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_12h[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_12h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals