#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12834_1h_4h_1d_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26
VOLUME_MA_PERIOD = 24
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA trend
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=EMA_FAST_PERIOD, adjust=False, min_periods=EMA_FAST_PERIOD).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=EMA_SLOW_PERIOD, adjust=False, min_periods=EMA_SLOW_PERIOD).mean().values
    ema_4h_trend = ema_4h_fast - ema_4h_slow  # Positive = bullish
    
    # Calculate 1d EMA trend
    close_1d = df_1d['close'].values
    ema_1d_fast = pd.Series(close_1d).ewm(span=EMA_FAST_PERIOD, adjust=False, min_periods=EMA_FAST_PERIOD).mean().values
    ema_1d_slow = pd.Series(close_1d).ewm(span=EMA_SLOW_PERIOD, adjust=False, min_periods=EMA_SLOW_PERIOD).mean().values
    ema_1d_trend = ema_1d_fast - ema_1d_slow  # Positive = bullish
    
    # Align to 1h timeframe
    ema_4h_trend_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_trend)
    ema_1d_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_trend)
    
    # Calculate 1h indicators
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_SLOW_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if EMA trends not available
        if np.isnan(ema_4h_trend_aligned[i]) or np.isnan(ema_1d_trend_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Combined trend filter (4h and 1d both agree)
        trend_bullish = ema_4h_trend_aligned[i] > 0 and ema_1d_trend_aligned[i] > 0
        trend_bearish = ema_4h_trend_aligned[i] < 0 and ema_1d_trend_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if volume_ok and trend_bullish:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif volume_ok and trend_bearish:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals