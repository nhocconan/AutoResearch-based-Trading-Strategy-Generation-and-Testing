#!/usr/bin/env python3
"""
Experiment #4627: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking 20-period Donchian channels with volume confirmation (>1.3x avg) 
in the direction of weekly pivot bias (price above/below weekly pivot) captures strong 
momentum in both bull and bear markets. Weekly pivot provides HTF bias to avoid counter-trend 
trades. Discrete sizing (0.25) and ATR trailing stop (2.5x) manage risk. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4627_6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot from prior week OHLC (avoid look-ahead)
    if len(df_1w) >= 1:
        # Prior week OHLC (shifted by 1)
        wh = np.concatenate([[np.nan], df_1w['high'].values[:-1]])  # prior week high
        wl = np.concatenate([[np.nan], df_1w['low'].values[:-1]])   # prior week low
        wc = np.concatenate([[np.nan], df_1w['close'].values[:-1]]) # prior week close
        wp = (wh + wl + wc) / 3.0  # weekly pivot point
    else:
        wp = np.array([])
    
    # Align weekly pivot to 6h timeframe
    if len(wp) > 0:
        wp_aligned = align_htf_to_ltf(prices, df_1w, wp)
    else:
        wp_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper channel: 20-period high
    # Lower channel: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(wp_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.3x average volume)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Weekly pivot bias: long bias if price > weekly pivot, short bias if price < weekly pivot
        long_bias = price > wp_aligned[i]
        short_bias = price < wp_aligned[i]
        
        # Donchian breakout conditions with volume confirmation and weekly pivot bias
        breakout_long = price > donchian_high[i] and vol_confirm and long_bias
        breakout_short = price < donchian_low[i] and vol_confirm and short_bias
        
        # Entry logic
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals