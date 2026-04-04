#!/usr/bin/env python3
"""
Experiment #4667: 6h Camarilla Pivot Reversal + 1d Volume Spike Filter
HYPOTHESIS: At 6h timeframe, price reversing from Camarilla R3/S3 levels (1d) with volume confirmation 
captures mean reversion in ranging markets while avoiding false breakouts. Works in bull (fading overextended rallies) 
and bear (fading panic spikes). Volume spike filter ensures institutional participation. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4667_6h_camarilla_pivot_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (using prior day's OHLC) ===
    if len(df_1d) >= 1:
        # Prior day's OHLC (shifted by 1 for completed bar only)
        prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
        prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
        prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
        
        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        # Camarilla levels
        rang = prev_high - prev_low
        r3 = pivot + rang * 1.1 / 2.0
        s3 = pivot - rang * 1.1 / 2.0
        r4 = pivot + rang * 1.1
        s4 = pivot - rang * 1.1
    else:
        pivot = np.full(len(df_1d), np.nan)
        r3 = np.full(len(df_1d), np.nan)
        s3 = np.full(len(df_1d), np.nan)
        r4 = np.full(len(df_1d), np.nan)
        s4 = np.full(len(df_1d), np.nan)
    
    # Align HTF indicators to 6h timeframe
    if len(r3) > 0:
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume Spike Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_spike = vol_ratio > 2.0  # Volume > 2x MA
    
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
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Fade at R3/S3 with volume spike confirmation
        fade_long = (price <= s3_aligned[i]) and vol_spike[i]
        fade_short = (price >= r3_aligned[i]) and vol_spike[i]
        
        if fade_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif fade_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals