#!/usr/bin/env python3
"""
Experiment #11979: 6h Price Channel Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: 6-hour price channel (Donchian) breakouts capture medium-term momentum. 
12-hour EMA provides trend bias to filter breakouts, reducing whipsaw. Volume confirmation 
ensures institutional participation. Works in bull markets (breakouts continue trend) 
and bear markets (breakouts fade quickly against trend). Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11979_6h_pricechannel_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CHANNEL_PERIOD = 20
TREND_EMA_PERIOD = 34
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_price_channel(high, low, period):
    """Calculate price channel (Donchian) - upper and lower bands"""
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
    # First value is simple average, then Wilder smoothing
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    # Set values before period to NaN for consistency
    atr[:period-1] = np.nan
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    ema_12h = calculate_ema(df_12h['close'].values, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Price channel (Donchian)
    channel_upper, channel_lower = calculate_price_channel(high, low, CHANNEL_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CHANNEL_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
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
        
        # Price channel breakout conditions (using previous bar's channel)
        breakout_up = (i > 0 and not np.isnan(channel_upper[i-1]) and 
                       high[i] > channel_upper[i-1])
        breakout_down = (i > 0 and not np.isnan(channel_lower[i-1]) and 
                         low[i] < channel_lower[i-1])
        
        # Volume confirmation
        volume_ok = (not np.isnan(volume_ma[i]) and 
                     volume[i] > (volume_ma[i] * VOLUME_THRESHOLD))
        
        # Trend filter (12h EMA)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_ok and uptrend_12h
        short_entry = breakout_down and volume_ok and downtrend_12h
        
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