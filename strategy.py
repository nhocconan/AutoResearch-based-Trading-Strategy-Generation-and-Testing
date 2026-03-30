#!/usr/bin/env python3
"""
Experiment #025: 6h Donchian + Volume + CHOP Regime Filter

HYPOTHESIS: Use 6h to catch momentum off-cycle from 4h/12h strategies.
CHOP < 50 as HARD filter (not loose) = only trade in trending regimes.
Donchian(20) breakout + volume confirm = proven edge from DB.
Size: 0.30.

WHY 6h: Captures momentum that fires between 4h/12h bars.
CHOP is the KEY differentiator - DB winners all use regime filtering.
Target: 60-120 total over 4 years per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_vol_chop_v1"
timeframe = "6h"
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

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (Ehlers): Measures market "choppiness"
    CHOP > 50 = ranging/consolidating (bad for breakout strategies)
    CHOP < 50 = trending (good for breakout strategies)
    
    Formula: 100 * log10(sum(ATR, period) / (highest ATR - lowest ATR)) / log10(period)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, 1)  # period=1 for raw TR
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr_sum = np.sum(atr[i - period + 1:i + 1])
        high_rolling = pd.Series(high[i - period + 1:i + 1]).max().values[0]
        low_rolling = pd.Series(low[i - period + 1:i + 1]).min().values[0]
        
        if high_rolling > low_rolling and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (high_rolling - low_rolling)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Pre-compute all indicators (vectorized) ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    
    # Donchian 20 (shift by 1 to avoid look-ahead)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === CHOP REGIME FILTER: CHOP < 50 = trending (trade), >= 50 = choppy (skip) ===
        # This is the KEY differentiator from previous failed strategies
        is_trending = chop[i] < 50.0
        
        # === HTF TREND ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === Donchian Breakout CONDITIONS ===
        bullish_breakout = (not np.isnan(dc_upper_20[i])) and (close[i] > dc_upper_20[i])
        bearish_breakout = (not np.isnan(dc_lower_20[i])) and (close[i] < dc_lower_20[i])
        
        # === VOLUME CONFIRMATION ===
        vol_ok = (vol_ma[i] > 1e-10) and (volume[i] > vol_ma[i] * 1.3)
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR from highest/lowest) ===
        if in_position:
            stop_hit = False
            if position_side > 0:
                # Long stop: price drops below highest - 2.5*ATR
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                # Short stop: price rises above lowest + 2.5*ATR
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on opposite HTF trend (after min hold)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        # KEY CHANGE: is_trending is now a HARD requirement
        if not in_position and is_trending:
            # LONG: Breakout above + volume confirm + 1d uptrend + trending regime
            if bullish_breakout and vol_ok and htf_bullish:
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakdown below + volume confirm + 1d downtrend + trending regime
            elif bearish_breakout and vol_ok and htf_bearish:
                in_position = True
                position_side = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals