#!/usr/bin/env python3
"""
Experiment #5219: 6h Camarilla Pivot Fade/Breakout + Volume Confirmation
HYPOTHESIS: On 6h timeframe, price reacts to 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout continuation) with volume confirmation (>1.5x average). Fade at R3/S3 in ranging markets, breakout at R4/S4 in trending markets. Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in bull/bear via adaptive logic: mean revert at extremes in chop, breakout with trend when volume confirms.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5219_6h_camarilla_pivot_vol_v1"
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
    
    # === 1d Indicators: Camarilla pivot levels ===
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla levels for each 1d bar
        camarilla_r4 = np.full(len(close_1d), np.nan)
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
        camarilla_s4 = np.full(len(close_1d), np.nan)
        camarilla_pp = np.full(len(close_1d), np.nan)
        
        for i in range(len(close_1d)):
            if i >= 1:  # Need previous day's data
                high_prev = high_1d[i-1]
                low_prev = low_1d[i-1]
                close_prev = close_1d[i-1]
                range_prev = high_prev - low_prev
                
                camarilla_pp[i] = (high_prev + low_prev + close_prev) / 3.0
                camarilla_r4[i] = camarilla_pp[i] + (range_prev * 1.5)
                camarilla_r3[i] = camarilla_pp[i] + (range_prev * 1.125)
                camarilla_s3[i] = camarilla_pp[i] - (range_prev * 1.125)
                camarilla_s4[i] = camarilla_pp[i] - (range_prev * 1.5)
        
        # Align to 6h timeframe (shifted by 1 for completed 1d bar only)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
        camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
        camarilla_pp_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Camarilla pivot conditions
        # Fade at R3/S3 (mean reversion) when price touches these levels
        # Breakout at R4/S4 (continuation) when price breaks these levels with volume
        fade_long = (price <= camarilla_s3_aligned[i] * 1.001) and (price >= camarilla_s3_aligned[i] * 0.999) and vol_confirm
        fade_short = (price >= camarilla_r3_aligned[i] * 0.999) and (price <= camarilla_r3_aligned[i] * 1.001) and vol_confirm
        breakout_long = (price >= camarilla_r4_aligned[i]) and vol_confirm
        breakout_short = (price <= camarilla_s4_aligned[i]) and vol_confirm
        
        # Final entry conditions
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