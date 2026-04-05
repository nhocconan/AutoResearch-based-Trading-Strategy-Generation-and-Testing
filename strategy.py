#!/usr/bin/env python3
"""
Experiment #10865: 12h Donchian Breakout with 1d Trend and Volume Confirmation
Hypothesis: 12-hour Donchian(20) breakouts in the direction of 1d EMA trend with volume confirmation provide high-probability trades.
Works in bull markets (trend following) and bear markets (mean reversion within trend).
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10865_12h_donchian_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_1D_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_high(low, high, period):
    """Calculate Donchian channel upper band"""
    return pd.Series(high).rolling(window=period, min_periods=period).max().values

def calculate_donchian_low(low, high, period):
    """Calculate Donchian channel lower band"""
    return pd.Series(low).rolling(window=period, min_periods=period).min().values

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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend
    ema_1d = calculate_ema(df_1d['close'].values, EMA_1D_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_high = calculate_donchian_high(low, high, DONCHIAN_PERIOD)
    donchian_low = calculate_donchian_low(low, high, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_1D_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 1d EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1] if not np.isnan(donchian_high[i-1]) else False
        breakout_down = close[i] < donchian_low[i-1] if not np.isnan(donchian_low[i-1]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_ok and uptrend_1d
        short_entry = breakout_down and volume_ok and downtrend_1d
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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