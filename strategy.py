#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d volume confirmation and 1d ATR filter
# Works in bull/bear by capturing breakouts with volume confirmation,
# using 1d ATR to normalize breakout strength and avoid false signals in low volatility.
# Target: 80-150 trades over 4 years (20-38/year) with controlled risk.

name = "exp_12871_6h_donchian20_1d_atr_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14
ATR_BREAKOUT_MULTIPLIER = 0.5  # Breakout must exceed 0.5*ATR
SIGNAL_SIZE = 0.25
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR for volatility normalization
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    
    # 1d volume moving average
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align 1d indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 6h Donchian channels
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_upper, donchian_lower = calculate_donchian(high_6h, low_6h, DONCHIAN_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 1d data not available
        if np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close_6h[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close_6h[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation: 1d volume > 1.5x MA
        volume_ok = volume_1d[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Breakout strength: price must exceed Donchian level by 0.5*1d ATR
        breakout_strength = atr_1d_aligned[i] * ATR_BREAKOUT_MULTIPLIER if not np.isnan(atr_1d_aligned[i]) else 0
        
        # Long breakout: close > upper + strength with volume
        breakout_long = volume_ok and (close_6h[i] > donchian_upper[i] + breakout_strength)
        
        # Short breakout: close < lower - strength with volume
        breakout_short = volume_ok and (close_6h[i] < donchian_lower[i] - breakout_strength)
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_6h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_6h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_1d_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals