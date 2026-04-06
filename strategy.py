#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12679_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # daily OHLC
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
    """
    Calculate Camarilla pivot levels for the day
    Based on previous day's H, L, C
    Returns levels: H4, L4, H3, L3, H2, L2, H1, L1
    """
    # Typical price
    pp = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    h4 = pp + range_ * 1.1 / 2
    l4 = pp - range_ * 1.1 / 2
    h3 = pp + range_ * 1.1 / 4
    l3 = pp - range_ * 1.1 / 4
    h2 = pp + range_ * 1.1 / 6
    l2 = pp - range_ * 1.1 / 6
    h1 = pp + range_ * 1.1 / 12
    l1 = pp - range_ * 1.1 / 12
    
    return h4, l4, h3, l3, h2, l2, h1, l1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # We use shift(1) to ensure we only use completed daily bars
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla for each day (using previous day's data)
    h4_1d = np.full_like(close_1d, np.nan)
    l4_1d = np.full_like(close_1d, np.nan)
    h3_1d = np.full_like(close_1d, np.nan)
    l3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        h4, l4, h3, l3, _, _, _, _ = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        h4_1d[i] = h4
        l4_1d[i] = l4
        h3_1d[i] = h3
        l3_1d[i] = l3
    
    # Align to 6h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
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
        # Skip if daily Camarilla not available
        if np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]):
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
        
        # Fade at H3/L3, breakout at H4/L4
        fade_long = close[i] <= l3_1d_aligned[i] and volume_ok
        fade_short = close[i] >= h3_1d_aligned[i] and volume_ok
        breakout_long = close[i] >= h4_1d_aligned[i] and volume_ok
        breakout_short = close[i] <= l4_1d_aligned[i] and volume_ok
        
        # Entry conditions
        long_entry = fade_long or breakout_long
        short_entry = fade_short or breakout_short
        
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