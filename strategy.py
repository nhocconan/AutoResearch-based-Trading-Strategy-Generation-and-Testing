#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation
# Works in bull/bear because breakouts capture strong moves, weekly trend filter ensures
# we trade in direction of higher timeframe momentum, and volume filters false breakouts.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and cost.

name = "exp_12997_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
TREND_PERIOD = 50

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian channels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    volume_d = df_daily['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian_channels(high_d, low_d, DONCHIAN_PERIOD)
    ema_trend = calculate_ema(close_d, TREND_PERIOD)
    volume_ma = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_d = calculate_atr(high_d, low_d, close_d, ATR_PERIOD)
    
    # Align to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, df_daily, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, df_daily, donchian_lower)
    ema_trend_4h = align_htf_to_ltf(prices, df_daily, ema_trend)
    volume_ma_4h = align_htf_to_ltf(prices, df_daily, volume_ma)
    atr_4h = align_htf_to_ltf(prices, df_daily, atr_d)
    close_d_4h = align_htf_to_ltf(prices, df_daily, close_d)
    
    # Calculate 4h indicators
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    volume_ma_4h_actual = pd.Series(volume_4h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_4h_actual = calculate_atr(high_4h, low_4h, close_4h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, TREND_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or \
           np.isnan(ema_trend_4h[i]) or np.isnan(volume_ma_4h[i]) or np.isnan(atr_4h[i]) or \
           np.isnan(close_d_4h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_4h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_4h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 4h volume)
        volume_ok = volume_4h[i] > (volume_ma_4h_actual[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_4h_actual[i]) else False
        
        # Trend filter: price above/below daily EMA
        uptrend = close_d_4h[i] > ema_trend_4h[i]
        downtrend = close_d_4h[i] < ema_trend_4h[i]
        
        # Breakout conditions
        breakout_long = volume_ok and uptrend and close_4h[i] >= donchian_upper_4h[i]
        breakout_short = volume_ok and downtrend and close_4h[i] <= donchian_lower_4h[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_4h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_4h_actual[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_4h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_4h_actual[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals