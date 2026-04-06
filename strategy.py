#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12495_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's high/low/close
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MIN_HOLD_BARS = 4  # Minimum 1 day hold (4 * 6h bars)

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
    """Calculate Camarilla levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return np.array([close, close, close, close, close, close, close, close])
    # Camarilla levels: Close +/- (range * multiplier)
    multipliers = [1.0/12, 1.0/6, 1.0/4, 1.0/2]  # L3, L2, L1, then H1, H2, H3, H4
    l3 = close - range_val * multipliers[0]
    l2 = close - range_val * multipliers[1]
    l1 = close - range_val * multipliers[2]
    h1 = close + range_val * multipliers[2]
    h2 = close + range_val * multipliers[1]
    h3 = close + range_val * multipliers[0]
    h4 = close + range_val * 0.55  # H4 uses 0.55 multiplier
    return np.array([l4, l3, l2, l1, h1, h2, h3, h4])  # Note: L4 not used in standard calc

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (using previous day's H/L/C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's levels (no look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First day will have invalid data (rolled from last), but we'll filter with valid index
    
    # Calculate Camarilla levels for previous day
    camarilla_levels = np.array([
        calculate_camarilla(high_1d_prev[i], low_1d_prev[i], close_1d_prev[i])
        for i in range(len(close_1d))
    ])
    
    # Extract levels: L3, L2, L1, H1, H2, H3
    l3 = camarilla_levels[:, 0] if camarilla_levels.size > 0 else np.full_like(close_1d, np.nan)
    l2 = camarilla_levels[:, 1] if camarilla_levels.size > 0 else np.full_like(close_1d, np.nan)
    l1 = camarilla_levels[:, 2] if camarilla_levels.size > 0 else np.full_like(close_1d, np.nan)
    h1 = camarilla_levels[:, 3] if camarilla_levels.size > 0 else np.full_like(close_1d, np.nan)
    h2 = camarilla_levels[:, 4] if camarilla_levels.size > 0 else np.full_like(close_1d, np.nan)
    h3 = camarilla_levels[:, 5] if camarilla_levels.size > 0 else np.full_like(close_1d, np.nan)
    
    # Align to 6h timeframe
    l3_6h = align_htf_to_ltf(prices, df_1d, l3)
    l2_6h = align_htf_to_ltf(prices, df_1d, l2)
    l1_6h = align_htf_to_ltf(prices, df_1d, l1)
    h1_6h = align_htf_to_ltf(prices, df_1d, h1)
    h2_6h = align_htf_to_ltf(prices, df_1d, h2)
    h3_6h = align_htf_to_ltf(prices, df_1d, h3)
    
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, 2) + 1  # Need at least 2 days for prev day data
    
    for i in range(start, n):
        # Skip if Camarilla levels not available (first day)
        if np.isnan(l1_6h[i]) or np.isnan(h1_6h[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Update bars since entry
        if position != 0:
            bars_since_entry += 1
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Fade at L3/H3 (mean reversion) or breakout at H4/L4 (but we use H3/L3 for fade)
        # Long near L3 (support), Short near H3 (resistance)
        long_fade = volume_ok and close[i] <= l3_6h[i] * 1.001  # Allow small buffer
        short_fade = volume_ok and close[i] >= h3_6h[i] * 0.999  # Allow small buffer
        
        # Entry conditions (only if minimum hold period satisfied or flat)
        long_entry = long_fade and (position != 1 or bars_since_entry >= MIN_HOLD_BARS)
        short_entry = short_fade and (position != -1 or bars_since_entry >= MIN_HOLD_BARS)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Consider reversing to short if strong short signal
            if short_entry and bars_since_entry >= MIN_HOLD_BARS:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Consider reversing to long if strong long signal
            if long_entry and bars_since_entry >= MIN_HOLD_BARS:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals