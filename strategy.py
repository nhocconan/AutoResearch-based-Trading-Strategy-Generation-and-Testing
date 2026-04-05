#!/usr/bin/env python3
"""
Experiment #10525: 12h Camarilla Pivot + Volume Spike + Chop Filter
Hypothesis: 12h Camarilla pivot levels from 1d provide strong support/resistance. 
When price touches these levels with volume expansion in a choppy market (Choppiness > 61.8),
mean reversion occurs. Works in both bull and bear markets as range-bound behavior persists.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10525_12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC
VOLUME_SPIKE_MULTIPLIER = 1.5
CHOPPINESS_PERIOD = 14
CHOPPINESS_THRESHOLD = 61.8  # >61.8 = choppy/range
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for next period"""
    range_ = high - low
    # Camarilla levels
    h5 = close + (range_ * 1.1 / 2)
    h4 = close + (range_ * 1.1 / 4)
    h3 = close + (range_ * 1.1 / 6)
    l3 = close - (range_ * 1.1 / 6)
    l4 = close - (range_ * 1.1 / 4)
    l5 = close - (range_ * 1.1 / 2)
    return h5, h4, h3, l3, l4, l5

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

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    range_max_min = highest_high - lowest_low
    
    # Avoid division by zero
    choppiness = np.where(
        range_max_min > 0,
        100 * np.log10(atr_sum / range_max_min) / np.log10(period),
        50  # neutral when no range
    )
    return choppiness

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from daily OHLC
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    h5, h4, h3, l3, l4, l5 = calculate_camarilla(daily_high, daily_low, daily_close)
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day for lookback)
    h5_aligned = align_htf_to_ltf(prices, df_daily, h5)
    h4_aligned = align_htf_to_ltf(prices, df_daily, h4)
    h3_aligned = align_htf_to_ltf(prices, df_daily, h3)
    l3_aligned = align_htf_to_ltf(prices, df_daily, l3)
    l4_aligned = align_htf_to_ltf(prices, df_daily, l4)
    l5_aligned = align_htf_to_ltf(prices, df_daily, l5)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Choppiness for regime filter
    chop = calculate_choppiness(high, low, close, CHOPPINESS_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, CHOPPINESS_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Chop filter: only trade in choppy/range market
        chop_filter = chop[i] > CHOPPINESS_THRESHOLD if not np.isnan(chop[i]) else False
        
        # Price proximity to Camarilla levels (within 0.1% of level)
        proximity_threshold = 0.001  # 0.1%
        near_h3 = abs(close[i] - h3_aligned[i]) / close[i] < proximity_threshold
        near_l3 = abs(close[i] - l3_aligned[i]) / close[i] < proximity_threshold
        near_h4 = abs(close[i] - h4_aligned[i]) / close[i] < proximity_threshold
        near_l4 = abs(close[i] - l4_aligned[i]) / close[i] < proximity_threshold
        
        # Entry conditions: touch Camarilla levels with volume in choppy market
        long_entry = (near_l3 or near_l4) and volume_spike and chop_filter
        short_entry = (near_h3 or near_h4) and volume_spike and chop_filter
        
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