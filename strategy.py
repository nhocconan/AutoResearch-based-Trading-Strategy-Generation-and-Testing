#!/usr/bin/env python3
"""
Experiment #8559: 6h Donchian breakout + 12h pivot reversal + volume confirmation.
Hypothesis: Combines trend-following breakouts (Donchian) with counter-trend reversals at extreme pivot levels (R4/S4) from higher timeframe.
Uses 12h pivots for context and volume to filter false breakouts. Works in both bull/bear by adapting to pivot levels.
Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee minimization.
"""

from mtf_data import get_alt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8559_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 10  # periods for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivots(high, low, close):
    """Calculate classic pivot points and support/resistance levels"""
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

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Initialize pivot arrays
    pivot_vals = np.full(len(close_12h), np.nan)
    r4_vals = np.full(len(close_12h), np.nan)
    s4_vals = np.full(len(close_12h), np.nan)
    
    # Calculate pivots for each 12h bar
    for i in range(len(close_12h)):
        p, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivots(high_12h[i], low_12h[i], close_12h[i])
        pivot_vals[i] = p
        r4_vals[i] = r4
        s4_vals[i] = s4
    
    # Align pivot data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_vals)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_vals)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_vals)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Pivot-based conditions
        near_r4 = close[i] >= r4_aligned[i] * 0.995  # Within 0.5% of R4
        near_s4 = close[i] <= s4_aligned[i] * 1.005  # Within 0.5% of S4
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry logic: 
        # - Fade at extreme pivot levels (R4/S4) with volume
        # - Continue breakout trend when not at extremes
        fade_long = near_s4 and volume_confirmed  # Buy near S4 with volume
        fade_short = near_r4 and volume_confirmed  # Sell near R4 with volume
        breakout_long = not near_r4 and long_breakout and volume_confirmed  # Breakout unless at R4
        breakout_short = not near_s4 and short_breakout and volume_confirmed  # Breakdown unless at S4
        
        # Combine signals
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