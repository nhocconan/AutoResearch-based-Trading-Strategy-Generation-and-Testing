#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h/1d EMA alignment and volume confirmation
# Uses 4h EMA for trend direction, 1d EMA for stronger trend filter, and volume spike for entry confirmation
# Works in bull/bear because trend following captures sustained moves, volume filters false breakouts,
# and multi-timeframe alignment reduces whipsaws. Target: 60-150 total trades over 4 years (15-37/year).

name = "exp_12994_1h_ema_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs on higher timeframes
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    ema_4h_fast = calculate_ema(close_4h, EMA_FAST_PERIOD)
    ema_4h_slow = calculate_ema(close_4h, EMA_SLOW_PERIOD)
    ema_1d_slow = calculate_ema(close_1d, EMA_SLOW_PERIOD)
    
    # Align to 1h timeframe
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    ema_1d_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slow)
    
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
    
    # Start from warmup period
    start = max(EMA_SLOW_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA data not available
        if np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]) or np.isnan(ema_1d_slow_aligned[i]):
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
        
        # Trend conditions: 4h fast > slow AND 1d > 4h slow (strong uptrend)
        # Or: 4h fast < slow AND 1d < 4h slow (strong downtrend)
        uptrend = ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i] and ema_1d_slow_aligned[i] > ema_4h_slow_aligned[i]
        downtrend = ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i] and ema_1d_slow_aligned[i] < ema_4h_slow_aligned[i]
        
        # Generate signals
        if position == 0:
            if uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif downtrend and volume_ok:
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