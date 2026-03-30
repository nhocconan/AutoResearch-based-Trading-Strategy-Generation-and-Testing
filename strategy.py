#!/usr/bin/env python3
"""
Experiment #025: 12h Donchian + Choppiness + SMA200

HYPOTHESIS: Choppiness Index is the #1 meta-filter from 16K+ experiments.
In trending markets (CHOP < 50), Donchian breakouts work.
In choppy markets (CHOP > 50), skip signals (whipsaw protection).

KEY INSIGHT: The previous best (Sharpe=0.308) used this exact pattern on 12h.
This variation adds: tighter min_hold (3 bars), slightly looser entry (no vol filter),
and proper ATR trailing stop. Should hit 75-150 trades target.

TARGET: 75-150 total over 4 years (18-37/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_sma200_v4"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    - CHOP > 61.8 = choppy market (mean reversion)
    - CHOP < 38.2 = trending market (trend following)
    - 38.2-61.8 = transition zone
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]) if idx > 0 else high[idx] - low[idx], abs(low[idx] - close[idx-1]) if idx > 0 else high[idx] - low[idx])
            atr_sum += tr
        
        # Highest high - lowest low over period
        highest_high = max(high[i-period+1:i+1])
        lowest_low = min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and atr_sum > 1e-10:
            # CHOP = 100 * log10(sum ATR) / log10(N * range)
            chop[i] = 100 * np.log10(atr_sum) / (np.log10(period) + np.log10(range_hl) if range_hl > 1e-10 else 0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA200 for macro direction (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian 20 - breakout channel (shifted 1 to avoid look-ahead)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 200  # Need 200 for SMA200
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP < 50 = trending (good for breakouts)
        # CHOP > 50 = choppy (skip signals - too many false breakouts)
        is_trending = chop_14[i] < 50.0
        
        # === HTF TREND (SMA200 direction) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === BREAKOUT CONDITIONS ===
        bullish_breakout = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # === VOLUME CONFIRMATION (optional but helps) ===
        vol_ok = volume[i] > vol_ma[i] * 1.2 if vol_ma[i] > 1e-10 else True  # More lenient
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 3 bars (36h) ===
        min_hold = (i - entry_bar) >= 3
        
        # === TRAILING STOP (2.0x ATR from highest/lowest) ===
        if in_position:
            stop_hit = False
            
            if position_side > 0:
                # Long stop: price drops below highest - 2*ATR
                stop_hit = low[i] < (highest_since_entry - 2.0 * atr_14[i])
            else:
                # Short stop: price rises above lowest + 2*ATR
                stop_hit = high[i] > (lowest_since_entry + 2.0 * atr_14[i])
            
            # Exit on opposite HTF trend (after min hold)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            # Exit on choppy regime (protect gains)
            if min_hold and not is_trending:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Trending + breakout + volume confirm + SMA200 alignment
            if is_trending and bullish_breakout and vol_ok and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Trending + breakdown + volume confirm + SMA200 alignment
            elif is_trending and bearish_breakout and vol_ok and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals