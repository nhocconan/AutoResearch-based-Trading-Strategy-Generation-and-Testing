#!/usr/bin/env python3
"""
Experiment #4709: 4h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter
HYPOTHESIS: 4h price breaking Donchian(20) channels with volume confirmation (>2.0x avg volume) and only in trending regimes (Choppiness Index < 38.2) captures strong momentum while avoiding whipsaws in ranging markets. The 1d timeframe provides volume context and regime filter works on 4h data. This strategy targets 19-50 trades/year on 4h timeframe to avoid fee drag while maintaining statistical significance. Works in both bull (breakouts with volume) and bear (short breakdowns with volume) markets by being directionally agnostic to trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4709_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for volume MA (using same symbol's 1d volume)
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF 1d Volume MA to 4h timeframe
    if len(vol_ma_1d) > 0:
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) from prior 20 bars ===
    # Use prior 20 bars' high/low (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 bars high
    pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 bars low
    
    # Rolling max/min of prior 20 bars
    donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Choppiness Index(14) for regime filter ===
    # Chop = 100 * log10(sum(ATR(1)/n) / (log10(HH - LL))) / log10(n)
    # where HH = highest high over n, LL = lowest low over n
    atr_1 = pd.Series(np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))).ewm(span=1, adjust=False).mean().values
    atr_1 = np.concatenate([[np.nan], atr_1])
    
    sum_atr_1 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and log of zero
    hl_range = hh - ll
    hl_range = np.where(hl_range <= 0, 1e-10, hl_range)
    
    chop = 100 * np.log10(sum_atr_1 / 14) / np.log10(14) / np.log10(hl_range)
    chop = np.where(hl_range <= 0, 50.0, chop)  # default to middle when range is zero
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # Donchian, ATR, Chop warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation for breakouts (>2.0x)
        vol_breakout = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_breakout
        breakout_short = price < donchian_low[i] and vol_breakout
        
        # Regime filter: only trade in trending markets (Chop < 38.2)
        regime_filter = chop[i] < 38.2
        
        # Final entry conditions: breakout + volume + regime filter
        if breakout_long and regime_filter:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and regime_filter:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals