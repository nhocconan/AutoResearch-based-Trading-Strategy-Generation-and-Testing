#!/usr/bin/env python3
"""
Experiment #10494: 1h Momentum + 4h Trend + 1d Volume Spike
Hypothesis: 1h momentum entries aligned with 4h trend and 1d volume confirmation
provide high-probability trend continuation trades. Uses 4h/1d for signal direction
and 1h only for entry timing to reduce noise. Target: 60-150 total trades over 4 years.
Works in bull markets (momentum with trend) and bear markets (mean reversion in range).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10494_1h_momentum_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
MOMENTUM_THRESHOLD = 0.02
TREND_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_momentum(close, period):
    """Calculate price momentum as percent change"""
    return pd.Series(close).pct_change(period).values

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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = calculate_ema(close_4h, TREND_PERIOD)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Pre-calculate session hours
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, TREND_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if 4h EMA not available
        if np.isnan(ema_4h_aligned[i]):
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
        
        # Volume spike confirmation (1d)
        volume_spike = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Trend filter: price above/below 4h EMA
        above_ema = close[i] > ema_4h_aligned[i]
        below_ema = close[i] < ema_4h_aligned[i]
        
        # Momentum conditions
        mom_up = momentum[i] > MOMENTUM_THRESHOLD if not np.isnan(momentum[i]) else False
        mom_down = momentum[i] < -MOMENTUM_THRESHOLD if not np.isnan(momentum[i]) else False
        
        # Entry conditions
        long_entry = mom_up and above_ema and volume_spike
        short_entry = mom_down and below_ema and volume_spike
        
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