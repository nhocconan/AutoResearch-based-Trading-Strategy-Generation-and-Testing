#!/usr/bin/env python3
"""
Experiment #4651: 6h Camarilla Pivot Fade/Breakout with 1d HTF + Volume Confirmation
HYPOTHESIS: 6h price fading at 1d Camarilla R3/S3 levels or breaking R4/S4 with volume confirmation captures mean reversion in ranging markets and continuation in trending markets. Uses discrete sizing (0.25) and ATR trailing stop (2.0x). Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4651_6h_camarilla_pivot_v1"
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
    
    # Calculate Camarilla levels from prior 1d OHLC (shifted by 1 to avoid look-ahead)
    if len(df_1d) >= 1:
        # Use prior day's OHLC (shifted by 1)
        ph = np.concatenate([[np.nan], df_1d['high'].values[:-1]])  # prior day high
        pl = np.concatenate([[np.nan], df_1d['low'].values[:-1]])   # prior day low
        pc = np.concatenate([[np.nan], df_1d['close'].values[:-1]]) # prior day close
        
        # Calculate Camarilla levels
        rng = ph - pl
        camarilla_h5 = pc + (rng * 1.1 / 2)  # R4
        camarilla_h4 = pc + (rng * 1.1 / 4)  # R3
        camarilla_h3 = pc + (rng * 1.1 / 6)  # R2
        camarilla_l3 = pc - (rng * 1.1 / 6)  # S2
        camarilla_l4 = pc - (rng * 1.1 / 4)  # S3
        camarilla_l5 = pc - (rng * 1.1 / 2)  # S4
    else:
        camarilla_h5 = np.full(n, np.nan)
        camarilla_h4 = np.full(n, np.nan)
        camarilla_h3 = np.full(n, np.nan)
        camarilla_l3 = np.full(n, np.nan)
        camarilla_l4 = np.full(n, np.nan)
        camarilla_l5 = np.full(n, np.nan)
    
    # Align Camarilla levels to 6h timeframe
    if len(camarilla_h5) > 0:
        h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    else:
        h5_aligned = np.full(n, np.nan)
        h4_aligned = np.full(n, np.nan)
        h3_aligned = np.full(n, np.nan)
        l3_aligned = np.full(n, np.nan)
        l4_aligned = np.full(n, np.nan)
        l5_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
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
        # Volume filter: confirmation for breakouts/fades (>1.3x)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Fade at R3/S3 (mean reversion in ranging markets)
        fade_long = price < h4_aligned[i] and price > l4_aligned[i] and vol_confirm
        fade_short = price > l4_aligned[i] and price < h4_aligned[i] and vol_confirm
        
        # Breakout at R4/S4 (continuation in trending markets)
        breakout_long = price > h5_aligned[i] and vol_confirm
        breakout_short = price < l5_aligned[i] and vol_confirm
        
        # Combine conditions: prefer fade in range, breakout in trend
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
        elif breakout_long:
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