#!/usr/bin/env python3
"""
Experiment #9574: 1h Donchian Breakout + Volume Spike + 4h/1d Trend Filter
Hypothesis: On 1h timeframe, use 4h and 1d trends for direction (avoiding counter-trend trades),
and 1h Donchian breakouts with volume confirmation for entry. This reduces whipsaw by
trading only with higher timeframe momentum. Targets 60-150 total trades over 4 years.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9574_1h_donchian_breakout_vol_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
TREND_EMA_FAST = 9
TREND_EMA_SLOW = 21
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h and 1d EMA trends
    close_4h = df_4h['close'].values
    ema_4h_fast = calculate_ema(close_4h, TREND_EMA_FAST)
    ema_4h_slow = calculate_ema(close_4h, TREND_EMA_SLOW)
    trend_4h = ema_4h_fast - ema_4h_slow  # >0 = uptrend, <0 = downtrend
    
    close_1d = df_1d['close'].values
    ema_1d_fast = calculate_ema(close_1d, TREND_EMA_FAST)
    ema_1d_slow = calculate_ema(close_1d, TREND_EMA_SLOW)
    trend_1d = ema_1d_fast - ema_1d_slow  # >0 = uptrend, <0 = downtrend
    
    # Align trends to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20, TREND_EMA_SLOW, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if trend data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine trend direction (both 4h and 1d must agree)
        bullish = trend_4h_aligned[i] > 0 and trend_1d_aligned[i] > 0
        bearish = trend_4h_aligned[i] < 0 and trend_1d_aligned[i] < 0
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout signals
        breakout_long = bullish and volume_spike and high[i] > highest_high[i]
        breakdown_short = bearish and volume_spike and low[i] < lowest_low[i]
        
        # Entry conditions
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakdown_short:
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