#!/usr/bin/env python3
"""
exp_7035_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Use 1w Camarilla pivots (calculated from prior week OHLC) to determine bias: long only above weekly H5, short only below weekly L5.
Volume confirms breakout legitimacy. Designed for 6h timeframe to capture swings with ~12-37 trades/year.
Works in both bull and bear markets by aligning with weekly structure - avoids counter-trend trades at key weekly levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7035_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 30  # ~7.5 months (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Camarilla pivot levels (H5, L5, H4, L4, H3, L3)
    # Based on prior week OHLC: H5 = HIGH + 1.1*(HIGH-LOW)/2, L5 = LOW - 1.1*(HIGH-LOW)/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels from prior week
    weekly_range = high_1w - low_1w
    h5 = high_1w + 1.1 * weekly_range / 2
    l5 = low_1w - 1.1 * weekly_range / 2
    h4 = high_1w + 1.1 * weekly_range / 4
    l4 = low_1w - 1.1 * weekly_range / 4
    h3 = high_1w + 1.1 * weekly_range / 6
    l3 = low_1w - 1.1 * weekly_range / 6
    
    # Align to LTF (6h) - shift(1) built into align_htf_to_ltf
    h5_aligned = align_htf_to_ltf(prices, df_1w, h5)
    l5_aligned = align_htf_to_ltf(prices, df_1w, l5)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]):
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
        
        # Determine bias from weekly Camarilla levels
        # Long bias: price above weekly H5 (strong bullish)
        # Short bias: price below weekly L5 (strong bearish)
        # No bias: price between H5 and L5 (wait for clearer signal)
        long_bias = close[i] > h5_aligned[i]
        short_bias = close[i] < l5_aligned[i]
        
        # Breakout signals aligned with weekly bias
        long_breakout = long_bias and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = short_bias and (close[i] < lowest_low[i]) and vol_confirmed
        
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