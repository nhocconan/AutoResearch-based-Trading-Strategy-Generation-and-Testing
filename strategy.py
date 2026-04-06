#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian channel (20) breakout with 12-hour volume confirmation and 1-day volatility regime filter.
# Breakouts from Donchian channels capture momentum moves. Volume confirmation ensures institutional participation.
# Volatility regime filter (using 1-day ATR ratio) avoids whipsaws in low-volatility environments and increases
# signal reliability. Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 50-150 total trades over 4 years.

name = "exp_13399_6h_donchian20_12h_vol_1d_volregime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 0.5  # ATR ratio > 0.5 of 20-period average
SIGNAL_SIZE = 0.25
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
    
    # Load 12h data ONCE before loop for volume
    df_12h = get_htf_data(prices, '12h')
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h volume MA
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Calculate 1d ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, VOLATILITY_LOOKBACK)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d  # Current ATR vs average ATR
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # 6h ATR for stoploss
    atr_6h = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, VOLATILITY_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(volume_ma_12h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr_6h[i]):
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
        
        # Volume confirmation (12h)
        volume_ok = volume[i] > (volume_ma_12h_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_12h_aligned[i]) else False
        
        # Volatility regime filter (1d): only trade when volatility is elevated
        vol_regime_ok = atr_ratio_1d_aligned[i] > VOLATILITY_THRESHOLD if not np.isnan(atr_ratio_1d_aligned[i]) else False
        
        # Breakout signals using Donchian channels
        breakout_up = volume_ok and vol_regime_ok and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and vol_regime_ok and (low[i] < lowest_low[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_6h[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_6h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals