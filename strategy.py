#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12554_1h_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h indicators
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    upper_4h, lower_4h = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    volume_ma_4h = pd.Series(volume_4h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, ATR_PERIOD)
    
    # Align 4h indicators to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 1h indicators for entry timing
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    upper_1h, lower_1h = calculate_donchian(high_1h, low_1h, DONCHIAN_PERIOD)
    volume_ma_1h = pd.Series(volume_1h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_1h = calculate_atr(high_1h, low_1h, close_1h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if daily EMA not available
        if np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_1h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_1h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 1h volume)
        volume_ok = volume_1h[i] > (volume_ma_1h[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1h[i]) else False
        
        # Trend filter (daily)
        uptrend_1d = close_1h[i] > ema_1d_aligned[i]
        downtrend_1d = close_1h[i] < ema_1d_aligned[i]
        
        # Donchian breakout conditions (using 1h for entry timing)
        long_breakout = close_1h[i] > upper_1h[i-1]  # break above previous upper band
        short_breakout = close_1h[i] < lower_1h[i-1]  # break below previous lower band
        
        # Entry conditions
        long_entry = volume_ok and uptrend_1d and long_breakout
        short_entry = volume_ok and downtrend_1d and short_breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_1h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1h[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_1h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals