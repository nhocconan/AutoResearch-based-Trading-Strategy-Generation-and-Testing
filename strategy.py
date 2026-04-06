#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with daily volume confirmation and weekly ATR filter
# Works in bull/bear by capturing strong momentum moves, filtering weak signals with volume,
# and adapting stoploss to volatility. Target: 100-200 trades over 4 years (25-50/year).

name = "exp_12917_4h_donchian20_1d_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
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
    
    # Load daily data ONCE before loop for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily indicators
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    volume_d = df_1d['volume'].values
    
    volume_ma = pd.Series(volume_d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr_d = calculate_atr(high_d, low_d, close_d, ATR_PERIOD)
    
    # Align daily indicators to 4h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_d)
    
    # Calculate 4h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_aligned[i]) else False
        
        # Donchian breakout
        breakout_long = volume_ok and (i >= DONCHIAN_PERIOD) and (high[i] >= donchian_upper[i-1])
        breakout_short = volume_ok and (i >= DONCHIAN_PERIOD) and (low[i] <= donchian_lower[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_aligned[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_aligned[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals