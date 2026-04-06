#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12559_6d_camarilla1d_v6"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Multiplier for Camarilla width
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
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
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    # Typical price
    typical = (high + low + close) / 3.0
    # Range
    range_val = high - low
    # Camarilla levels
    h4 = typical + (range_val * 1.1 * CAMARILLA_MULT / 2)
    h3 = typical + (range_val * 1.1 / 2)
    l3 = typical - (range_val * 1.1 / 2)
    l4 = typical - (range_val * 1.1 * CAMARILLA_MULT / 2)
    return h4, h3, l3, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility filter
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, ATR_PERIOD)
    atr_1d_ma = pd.Series(atr_1d).rolling(window=5, min_periods=5).mean().values  # 5-day ATR MA
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # Calculate daily Camarilla levels
    h4_1d, h3_1d, l3_1d, l4_1d = calculate_camarilla(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
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
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, 5) + 1  # 5 for ATR MA
    
    for i in range(start, n):
        # Skip if daily ATR MA not available
        if np.isnan(atr_1d_ma_aligned[i]):
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
        
        # Volatility filter: only trade when volatility is elevated (above 5-day MA)
        vol_filter = atr[i] > atr_1d_ma_aligned[i] if not np.isnan(atr_1d_ma_aligned[i]) else False
        
        # Camarilla fade conditions (mean reversion at extreme levels)
        fade_long = close[i] <= l4_1d_aligned[i]  # Price at or below L4 (extreme low)
        fade_short = close[i] >= h4_1d_aligned[i]  # Price at or above H4 (extreme high)
        
        # Entry conditions: fade extremes with volume and volatility confirmation
        long_entry = volume_ok and vol_filter and fade_long
        short_entry = volume_ok and vol_filter and fade_short
        
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