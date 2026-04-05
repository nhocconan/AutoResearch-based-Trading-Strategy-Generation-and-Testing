#!/usr/bin/env python3
"""
Experiment #11255: 6h Volume Spike + Weekly Support/Resistance
Hypothesis: Volume spikes at key weekly levels indicate strong directional moves. 
In bull markets, volume spikes break weekly resistance; in bear markets, 
volume spikes break weekly support. Uses 1w levels for structure and 6h for timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11255_6h_vol_spike_weekly_sr_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 2.0
WEEKLY_LOOKBACK = 12  # ~3 months for weekly high/low
SUPPORT_RESISTANCE_BUFFER = 0.002  # 0.2% buffer
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_weekly_high_low(prices):
    """Calculate weekly high and low from daily data"""
    # Resample to weekly using actual weekly data from parquet
    weekly_data = get_htf_data(prices, '1w')
    if len(weekly_data) < 2:
        return np.full(len(prices), np.nan), np.full(len(prices), np.nan)
    
    weekly_high = weekly_data['high'].values
    weekly_low = weekly_data['low'].values
    
    # Expand rolling window for weekly high/low
    roll_high = pd.Series(weekly_high).rolling(window=WEEKLY_LOOKBACK, min_periods=1).max().values
    roll_low = pd.Series(weekly_low).rolling(window=WEEKLY_LOOKBACK, min_periods=1).min().values
    
    # Align to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, weekly_data, roll_high)
    weekly_low_aligned = align_htf_to_ltf(prices, weekly_data, roll_low)
    
    return weekly_high_aligned, weekly_low_aligned

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Weekly support/resistance
    weekly_high, weekly_low = calculate_weekly_high_low(prices)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly levels not available
        if np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]):
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
        
        # Volume spike condition
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Distance to weekly levels
        dist_to_high = (weekly_high[i] - close[i]) / close[i] if weekly_high[i] > 0 else 1
        dist_to_low = (close[i] - weekly_low[i]) / close[i] if weekly_low[i] > 0 else 1
        
        # Near weekly resistance (within buffer) - potential breakout
        near_resistance = dist_to_high < SUPPORT_RESISTANCE_BUFFER and dist_to_high > 0
        # Near weekly support (within buffer) - potential breakdown
        near_support = dist_to_low < SUPPORT_RESISTANCE_BUFFER and dist_to_low > 0
        
        # Entry conditions
        long_entry = volume_spike and near_resistance and close[i] > weekly_high[i]
        short_entry = volume_spike and near_support and close[i] < weekly_low[i]
        
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