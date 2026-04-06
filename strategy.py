#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13914_1h_4h_donchian20_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian(high, low, period):
    """Calculate Donchian upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend direction
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_TREND_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data for Donchian, ATR, and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels on 4h
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, DONCHIAN_PERIOD)
    
    # ATR on 4h
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, ATR_PERIOD)
    
    # Volume confirmation on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using 4h volume aligned to 1h)
        volume_ok = volume[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD)
        
        # Trend filter from 1d EMA
        trend_up = close[i] > ema_1d_aligned[i]
        trend_down = close[i] < ema_1d_aligned[i]
        
        # Donchian breakout signals on 4h (aligned to 1h)
        breakout_up = close[i] > donchian_upper_aligned[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower_aligned[i-1]  # break below previous lower band
        
        # Entry signals
        long_signal = volume_ok and trend_up and breakout_up
        short_signal = volume_ok and trend_down and breakout_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_aligned[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or trend change
            if close[i] < donchian_lower_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on Donchian breakout or trend change
            if close[i] > donchian_upper_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals