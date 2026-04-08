#!/usr/bin/env python3
"""
Experiment #4615: 6h Camarilla Pivot Breakout/Fade + Volume Confirmation
HYPOTHESIS: 6h price breaking Camarilla R4/S4 levels (from prior 1d) with volume confirmation (>1.5x avg) captures strong momentum breakouts, while fading R3/S3 levels with volume exhaustion (<0.7x avg) captures mean reversion in ranging markets. Uses 1d HTF for pivot calculation to avoid look-ahead. Discrete sizing (0.25) and ATR trailing stop (2.0x) manage risk. Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4615_6h_camarilla_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # S4 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.1/4), etc.
    if len(df_1d) >= 1:
        # Use prior day's OHLC (shifted by 1 to avoid look-ahead)
        ph = np.concatenate([[np.nan], df_1d['high'].values[:-1]])  # prior day high
        pl = np.concatenate([[np.nan], df_1d['low'].values[:-1]])   # prior day low
        pc = np.concatenate([[np.nan], df_1d['close'].values[:-1]]) # prior day close
        
        # Camarilla calculations
        rng = ph - pl
        camarilla_r4 = pc + (rng * 1.1 / 2.0)
        camarilla_r3 = pc + (rng * 1.1 / 4.0)
        camarilla_s3 = pc - (rng * 1.1 / 4.0)
        camarilla_s4 = pc - (rng * 1.1 / 2.0)
    else:
        camarilla_r4 = np.array([])
        camarilla_r3 = np.array([])
        camarilla_s3 = np.array([])
        camarilla_s4 = np.array([])
    
    # Align Camarilla levels to 6h timeframe
    if len(camarilla_r4) > 0:
        r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filters: confirmation for breakouts (>1.5x), exhaustion for fades (<0.7x)
        vol_breakout = vol_ratio[i] > 1.5   # Strong volume for breakout
        vol_exhaustion = vol_ratio[i] < 0.7 # Weak volume for fade
        
        # Breakout conditions: price breaks R4/S4 with volume confirmation
        breakout_long = price > r4_aligned[i] and vol_breakout
        breakout_short = price < s4_aligned[i] and vol_breakout
        
        # Fade conditions: price rejects R3/S3 with volume exhaustion
        fade_long = price < r3_aligned[i] and price > s3_aligned[i] and vol_exhaustion
        fade_short = price > r3_aligned[i] and price < s3_aligned[i] and vol_exhaustion
        
        # Combine logic: breakouts take precedence, fades only in range
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
        elif fade_long and not (breakout_long or breakout_short):
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif fade_short and not (breakout_long or breakout_short):
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals