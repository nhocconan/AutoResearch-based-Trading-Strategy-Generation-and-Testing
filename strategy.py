#!/usr/bin/env python3
"""
Experiment #4855: 6h Donchian(20) Breakout + 1w Camarilla Pivot + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly Camarilla pivot levels (R3/S3 for continuation, R4/S4 for strong breakouts) with volume confirmation (>1.5x average) capture sustained momentum. Weekly pivot provides structural support/resistance that works in both bull (breakouts above R3) and bear (breakdowns below S3) markets. Target: 12-37 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4855_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Camarilla Pivot Levels (using previous week's OHLC) ===
    if len(df_1w) >= 1:
        # Calculate pivot points from previous week's OHLC
        # We need to shift by 1 week to avoid look-ahead (use completed week only)
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Previous week's values (shifted by 1)
        prev_high = np.concatenate([[np.nan], high_1w[:-1]])
        prev_low = np.concatenate([[np.nan], low_1w[:-1]])
        prev_close = np.concatenate([[np.nan], close_1w[:-1]])
        
        # Camarilla pivot calculation
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
        s3 = pivot - (range_val * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
        r4 = pivot + (range_val * 1.1)        # R4 = pivot + 1.1*(H-L)
        s4 = pivot - (range_val * 1.1)        # S4 = pivot - 1.1*(H-L)
        
        # Align to 6h timeframe (use align_htf_to_ltf which includes shift(1))
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with Camarilla pivot alignment
        # Long: break above Donchian high AND above S3 (bullish continuation) OR above R4 (strong breakout)
        breakout_long = (
            (price >= high_roll[i]) and 
            vol_confirm and
            ((price > s3_aligned[i]) or (price > r4_aligned[i]))
        )
        
        # Short: break below Donchian low AND below R3 (bearish continuation) OR below S4 (strong breakdown)
        breakout_short = (
            (price <= low_roll[i]) and 
            vol_confirm and
            ((price < r3_aligned[i]) or (price < s4_aligned[i]))
        )
        
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