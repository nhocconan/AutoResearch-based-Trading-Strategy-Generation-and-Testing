#!/usr/bin/env python3
"""
exp_7092_12h_camarilla1d_pivot_vol_v1
Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR stoploss.
Goes long when price breaks above H3/H4 with volume spike, short when breaks below L3/L4.
Uses 1d Camarilla levels (proven ETH edge) on 12h timeframe to capture swings with 12-37 trades/year.
Volume confirms breakout legitimacy. Works in both bull/bear markets by following price action.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7092_12h_camarilla1d_pivot_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 25  # ~25 * 12h = 12.5 days

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H4, H3, L3, L4
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h4 = close_1d + range_1d * 1.1 / 2
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    l4 = close_1d - range_1d * 1.1 / 2
    
    # Align to LTF (12h)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Camarilla breakout signals
        long_breakout = vol_confirmed and close[i] > h4_aligned[i]
        short_breakout = vol_confirmed and close[i] < l4_aligned[i]
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals