#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1d trend filter and volume confirmation
# Works in bull/bear because breakouts capture strong moves, volume filters weak signals,
# and daily trend filter prevents counter-trend entries. Target: 100-200 trades over 4 years (25-50/year).

name = "exp_12880_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_FAST = 9
TREND_EMA_SLOW = 21
VOLUME_MA_PERIOD = 20
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
    
    # Donchian channels on daily
    donch_upper, donch_lower = calculate_donchian(high_d, low_d, DONCHIAN_PERIOD)
    
    # EMA trend on daily
    ema_fast = calculate_ema(close_d, TREND_EMA_FAST)
    ema_slow = calculate_ema(close_d, TREND_EMA_SLOW)
    
    # Volume MA on daily
    volume_ma = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR on daily for stoploss
    atr_d = calculate_atr(high_d, low_d, close_d, ATR_PERIOD)
    
    # Align daily indicators to 4h timeframe
    donch_upper_4h = align_htf_to_ltf(prices, df_daily, donch_upper)
    donch_lower_4h = align_htf_to_ltf(prices, df_daily, donch_lower)
    ema_fast_4h = align_htf_to_ltf(prices, df_daily, ema_fast)
    ema_slow_4h = align_htf_to_ltf(prices, df_daily, ema_slow)
    volume_ma_4h = align_htf_to_ltf(prices, df_daily, volume_ma)
    atr_4h = align_htf_to_ltf(prices, df_daily, atr_d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(donch_upper_4h[i]) or np.isnan(donch_lower_4h[i]) or \
           np.isnan(ema_fast_4h[i]) or np.isnan(ema_slow_4h[i]) or \
           np.isnan(volume_ma_4h[i]) or np.isnan(atr_4h[i]):
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
        volume_ok = volume[i] > (volume_ma_4h[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_4h[i]) else False
        
        # Trend filter
        uptrend = ema_fast_4h[i] > ema_slow_4h[i]
        downtrend = ema_fast_4h[i] < ema_slow_4h[i]
        
        # Breakout conditions
        breakout_long = volume_ok and uptrend and close[i] >= donch_upper_4h[i]
        breakout_short = volume_ok and downtrend and close[i] <= donch_lower_4h[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_4h[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_4h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals