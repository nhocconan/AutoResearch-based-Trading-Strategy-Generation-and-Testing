#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price channels with 1-day volume confirmation and regime filter
# Works in bull/bear by capturing breakouts from established ranges, volume filters false signals,
# and choppiness regime filter ensures we only trade in trending conditions.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13012_12h_channel_breakout_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CHANNEL_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8  # Above 61.8 = choppy (range), below = trending
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

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    atr_sum = pd.Series(calculate_atr(high, low, close, 1)).rolling(window=period, min_periods=period).sum()
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
    return chop.values

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
    
    # 1d volume MA
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # 1d choppiness
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, CHOP_PERIOD)
    
    # Align to 12h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h price channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    upper_channel = pd.Series(high).rolling(window=CHANNEL_PERIOD, min_periods=CHANNEL_PERIOD).max().values
    lower_channel = pd.Series(low).rolling(window=CHANNEL_PERIOD, min_periods=CHANNEL_PERIOD).min().values
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CHANNEL_PERIOD, VOLUME_MA_PERIOD, CHOP_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
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
        
        # Volume confirmation (1d volume > 1.5x MA)
        volume_ok = volume[i] > (volume_ma_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_aligned[i]) else False
        
        # Regime filter: only trade when NOT choppy (trending market)
        trending = chop_aligned[i] < CHOP_THRESHOLD
        
        # Breakout signals
        breakout_long = volume_ok and trending and close[i] >= upper_channel[i]
        breakout_short = volume_ok and trending and close[i] <= lower_channel[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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