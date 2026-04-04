#!/usr/bin/env python3
"""
Experiment #4927: 6h Camarilla Pivot + 1d Volume Spike + ATR Filter
HYPOTHESIS: On 6h timeframe, Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe provide high-probability reversal/continuation zones. Volume confirmation (>2x average) filters false signals, and ATR(14) trailing stop (2.0x) manages risk. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts at R4/S4 with volume) and bear markets (mean reversion at R3/S3 with volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4927_6h_camarilla_pivot_1d_vol_atr_v1"
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
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3, R4, S4) ===
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate pivots for each 1d bar (using previous day's OHLC)
        pivot = np.full(len(df_1d), np.nan)
        r3 = np.full(len(df_1d), np.nan)
        s3 = np.full(len(df_1d), np.nan)
        r4 = np.full(len(df_1d), np.nan)
        s4 = np.full(len(df_1d), np.nan)
        
        for i in range(1, len(df_1d)):
            # Use previous day's OHLC to calculate today's pivot levels
            h = high_1d[i-1]
            l = low_1d[i-1]
            c = close_1d[i-1]
            
            pivot[i] = (h + l + c) / 3.0
            range_hl = h - l
            r3[i] = pivot[i] + range_hl * 1.1 / 2.0  # R3 = pivot + (H-L)*1.1/2
            s3[i] = pivot[i] - range_hl * 1.1 / 2.0  # S3 = pivot - (H-L)*1.1/2
            r4[i] = pivot[i] + range_hl * 1.1        # R4 = pivot + (H-L)*1.1
            s4[i] = pivot[i] - range_hl * 1.1        # S4 = pivot - (H-L)*1.1
    else:
        pivot = np.full(len(df_1d), np.nan)
        r3 = np.full(len(df_1d), np.nan)
        s3 = np.full(len(df_1d), np.nan)
        r4 = np.full(len(df_1d), np.nan)
        s4 = np.full(len(df_1d), np.nan)
    
    # Align HTF Camarilla levels to 6h timeframe
    if len(pivot) > 0:
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
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
    
    warmup = max(20, 20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Camarilla pivot conditions:
        # Mean reversion at R3/S3 (price touches extreme and reverses)
        # Breakout continuation at R4/S4 (price breaks extreme with volume)
        mean_revert_long = (price <= s3_aligned[i] * 1.001) and (price >= s3_aligned[i] * 0.999) and vol_confirm
        mean_revert_short = (price >= r3_aligned[i] * 0.999) and (price <= r3_aligned[i] * 1.001) and vol_confirm
        breakout_long = (price >= r4_aligned[i]) and vol_confirm
        breakout_short = (price <= s4_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if mean_revert_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif mean_revert_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals