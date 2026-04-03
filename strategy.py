#!/usr/bin/env python3
"""
Experiment #207: 6h Camarilla Pivot Reversal + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from daily timeframe act as institutional support/resistance. 
Reversals at R3/S3 with 1d volume spike and alignment with 1w EMA trend capture high-probability mean-reversion 
in ranging markets and continuation in strong trends. This works in both bull/bear markets by fading extremes 
in ranges and following the weekly trend. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_reversal_1d_1w_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF: 1w data for EMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels from previous 1d bar
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_h5 = prev_close + range_val * 1.1 / 2  # R4
        camarilla_h4 = prev_close + range_val * 1.1 / 4  # R3
        camarilla_l5 = prev_close - range_val * 1.1 / 2  # S4
        camarilla_l4 = prev_close - range_val * 1.1 / 4  # S3
        camarilla_h3 = prev_close + range_val * 1.1 / 6  # R2
        camarilla_l3 = prev_close - range_val * 1.1 / 6  # S2
        camarilla_h2 = prev_close + range_val * 1.1 / 12 # R1
        camarilla_l2 = prev_close - range_val * 1.1 / 12 # S1
        camarilla_h1 = prev_close + range_val * 1.1 / 24 # PP + small
        camarilla_l1 = prev_close - range_val * 1.1 / 24 # PP - small
        
        # Align to 6h timeframe (already shift(1) from prev bar)
        h5 = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        l5 = align_htf_to_ltf(prices, df_1d, camarilla_l5)
        l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        h5 = h4 = l5 = l4 = h3 = l3 = np.full(n, np.nan)
    
    # Calculate 1d volume MA(20) for spike detection
    if len(df_1d) >= 20:
        vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    else:
        vol_ma_20_aligned = np.full(n, np.nan)
    
    # Calculate 1w EMA(21) for trend filter
    if len(df_1w) >= 21:
        ema_21 = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    else:
        ema_21_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(h5[i]) or np.isnan(l5[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20_aligned[i] * 2.0 if vol_ma_20_aligned[i] > 1e-10 else False
        
        # --- Trend Filter (1w EMA) ---
        trend_bullish = close[i] > ema_21_aligned[i]
        trend_bearish = close[i] < ema_21_aligned[i]
        
        # --- Camarilla Levels Logic ---
        # Reversal at H4/L4 (R3/S3) with volume and trend alignment
        # Continuation break of H5/L5 (R4/S4) with volume
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: opposite Camarilla level touch or trend reversal
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches L4 (S3) OR breaks below 1w EMA
                    if close[i] <= l4[i] or close[i] < ema_21_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches H4 (R3) OR breaks above 1w EMA
                    if close[i] >= h4[i] or close[i] > ema_21_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long reversal at L4 (S3) with volume and bullish trend alignment
        if low[i] <= l4[i] and vol_ok and trend_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short reversal at H4 (R3) with volume and bearish trend alignment
        elif high[i] >= h4[i] and vol_ok and trend_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        # Long continuation break above H5 (R4) with volume
        elif high[i] > h5[i] and vol_ok and trend_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short continuation break below L5 (S4) with volume
        elif low[i] < l5[i] and vol_ok and trend_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals