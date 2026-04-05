#!/usr/bin/env python3
"""
Experiment #9757: 4h Donchian Breakout + Volume Spike + ATR Stop
Hypothesis: 4h Donchian(20) breakouts with volume confirmation provide
directional edge in both bull and bear markets. Volume filters false breakouts.
ATR-based stop manages risk. Targets 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9757_4h_donchian_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for context (not used in signals but available if needed)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    dc_upper, dc_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume MA for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD)
    
    for i in range(start, n):
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
        
        # Volume spike confirmation
        vol_ma = volume_ma[i]
        vol_spike = volume[i] > (vol_ma * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(vol_ma) else False
        
        # Breakout conditions
        breakout_up = high[i] >= dc_upper[i]  # Using high to catch breakout
        breakout_down = low[i] <= dc_lower[i]  # Using low to catch breakdown
        
        # Entry logic
        if position == 0:
            if breakout_up and vol_spike:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down and vol_spike:
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