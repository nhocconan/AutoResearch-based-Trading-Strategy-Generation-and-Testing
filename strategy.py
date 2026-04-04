#!/usr/bin/env python3
"""
Experiment #5195: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction capture institutional momentum. Weekly pivot (calculated from prior week's high/low/close) provides structural support/resistance that works in both bull and bear markets. Volume > 2.0x average confirms participation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5195_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly Pivot Points (using prior week's OHLC) ===
    if len(df_1w) >= 1:
        # Weekly pivot: P = (H + L + C) / 3
        # R1 = 2*P - L, S1 = 2*P - H
        # R2 = P + (H - L), S2 = P - (H - L)
        # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        r1_1w = 2 * pivot_1w - low_1w
        s1_1w = 2 * pivot_1w - high_1w
        r2_1w = pivot_1w + (high_1w - low_1w)
        s2_1w = pivot_1w - (high_1w - low_1w)
        r3_1w = high_1w + 2 * (pivot_1w - low_1w)
        s3_1w = low_1w - 2 * (high_1w - pivot_1w)
        
        # Align to 6h timeframe (shifted by 1 for completed weekly bars only)
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
        r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
        s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
        r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
        s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
        r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
        s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    else:
        pivot_1w_aligned = np.full(n, np.nan)
        r1_1w_aligned = np.full(n, np.nan)
        s1_1w_aligned = np.full(n, np.nan)
        r2_1w_aligned = np.full(n, np.nan)
        s2_1w_aligned = np.full(n, np.nan)
        r3_1w_aligned = np.full(n, np.nan)
        s3_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2.0x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Donchian breakout conditions with weekly pivot direction filter
        # Long: Donchian breakout above + price > weekly R1 (bullish bias)
        # Short: Donchian breakdown below + price < weekly S1 (bearish bias)
        breakout_long = (price >= high_roll[i]) and (price > r1_1w_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < s1_1w_aligned[i]) and vol_confirm
        
        # Final entry conditions
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