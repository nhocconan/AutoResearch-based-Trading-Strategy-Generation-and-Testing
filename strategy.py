#!/usr/bin/env python3
"""
exp_7475_6d_donchian20_1w_pivot_vol_v1
Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Uses weekly pivot levels (R4/S4) to determine primary trend direction, targeting 75-150 trades over 4 years.
Designed to work in both bull and bear markets by following weekly pivot structure.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7475_6d_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 10  # weeks to look back for pivot calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # 3 days for 6h timeframe

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points (standard method)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize arrays for pivot levels
    r4 = np.full_like(close_1w, np.nan)
    s4 = np.full_like(close_1w, np.nan)
    
    # Calculate pivot points for each week using lookback period
    for i in range(PIVOT_LOOKBACK, len(close_1w)):
        # Use PIVOT_LOOKBACK weeks of data to calculate current week's pivot
        start_idx = i - PIVOT_LOOKBACK + 1
        if start_idx >= 0:
            # Use the most recent week's data (current week)
            idx = i
            _, _, _, _, r4_val, _, _, _, s4_val = calculate_pivot_points(
                high_1w[idx], low_1w[idx], close_1w[idx]
            )
            r4[i] = r4_val
            s4[i] = s4_val
    
    # Align to LTF (6h)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Determine market regime based on weekly pivot levels
        above_r4 = close[i] > r4_aligned[i]
        below_s4 = close[i] < s4_aligned[i]
        
        # Breakout trades in direction of weekly pivot extremes
        breakout_long = above_r4 and (close[i] > highest_high[i]) and vol_confirmed
        breakout_short = below_s4 and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Mean reversion at pivot extremes (fade extreme moves)
        fade_long = below_s4 and (close[i] < lowest_low[i]) and vol_confirmed
        fade_short = above_r4 and (close[i] > highest_high[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
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