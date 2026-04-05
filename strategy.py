#!/usr/bin/env python3
"""
Experiment #9654: 1h EMA Pullback + 4h Trend + Volume + Session Filter.
Hypothesis: In trending markets (4h EMA alignment), pullbacks to the 21 EMA on 1h with volume confirmation provide high-probability entries. 
Session filter (08-20 UTC) reduces noise. Uses 4h for trend direction, 1h for entry timing. Targets 60-150 trades over 4 years.
Works in bull (long on uptrend pullbacks) and bear (short on downtrend pullbacks).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9654_1h_ema_pullback_4h_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMAs for trend
    close_4h = df_4h['close'].values
    ema_fast_4h = calculate_ema(close_4h, EMA_FAST)
    ema_slow_4h = calculate_ema(close_4h, EMA_SLOW)
    ema_trend_4h = calculate_ema(close_4h, EMA_TREND)
    
    # Align 4h EMAs to 1h
    ema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_fast_4h)
    ema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slow_4h)
    ema_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_trend_4h)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMAs
    ema_fast_1h = calculate_ema(close, EMA_FAST)
    ema_slow_1h = calculate_ema(close, EMA_SLOW)
    
    # Volume MA for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside session
        if not session_mask[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_fast_4h_aligned[i]) or np.isnan(ema_slow_4h_aligned[i]) or np.isnan(ema_trend_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
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
        
        # Trend direction from 4h EMA alignment
        # Uptrend: fast > slow > trend
        # Downtrend: fast < slow < trend
        uptrend_4h = ema_fast_4h_aligned[i] > ema_slow_4h_aligned[i] > ema_trend_4h_aligned[i]
        downtrend_4h = ema_fast_4h_aligned[i] < ema_slow_4h_aligned[i] < ema_trend_4h_aligned[i]
        
        # 1h EMA pullback conditions
        # Pullback to 21 EMA in direction of trend
        pullback_to_ema = (low[i] <= ema_slow_1h[i] <= high[i]) or (abs(close[i] - ema_slow_1h[i]) < ema_slow_1h[i] * 0.005)
        
        # Volume confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = uptrend_4h and pullback_to_ema and volume_spike
        short_entry = downtrend_4h and pullback_to_ema and volume_spike
        
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