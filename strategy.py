#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12539_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's data
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
CAMARILLA_MULT = 1.1  # Breakout threshold multiplier

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla levels from previous period"""
    range_val = high - low
    # Camarilla levels
    h5 = close + range_val * 1.1 / 2
    h4 = close + range_val * 1.1
    h3 = close + range_val * 1.1 * 1.16 / 2
    l3 = close - range_val * 1.1 * 1.16 / 2
    l4 = close - range_val * 1.1
    l5 = close - range_val * 1.1 / 2
    return h3, h4, h5, l3, l4, l5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, ATR_PERIOD)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    h3_1d = np.zeros(len(high_1d))
    h4_1d = np.zeros(len(high_1d))
    h5_1d = np.zeros(len(high_1d))
    l3_1d = np.zeros(len(high_1d))
    l4_1d = np.zeros(len(high_1d))
    l5_1d = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        h3, h4, h5, l3, l4, l5 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        h3_1d[i] = h3
        h4_1d[i] = h4
        h5_1d[i] = h5
        l3_1d[i] = l3
        l4_1d[i] = l4
        l5_1d[i] = l5
    
    # Align Camarilla levels to 6h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h5_1d_aligned = align_htf_to_ltf(prices, df_1d, h5_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    l5_1d_aligned = align_htf_to_ltf(prices, df_1d, l5_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily data not available
        if np.isnan(h3_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Camarilla breakout conditions (using previous day's levels)
        # Long: break above H4 with volume
        # Short: break below L4 with volume
        long_breakout = close[i] > (h4_1d_aligned[i] * CAMARILLA_MULT)
        short_breakout = close[i] < (l4_1d_aligned[i] / CAMARILLA_MULT)
        
        # Entry conditions
        long_entry = volume_ok and long_breakout
        short_entry = volume_ok and short_breakout
        
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