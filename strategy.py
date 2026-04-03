#!/usr/bin/env python3
"""
Experiment #095: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts in direction of weekly Camarilla pivot bias with volume confirmation (>1.5x) capture medium-term momentum. Weekly pivot provides structural bias (R3/S3 for mean reversion, R4/S4 for breakout) that works in both bull and bear markets. Uses discrete sizing (0.25) and ATR stoploss (2.0*ATR). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_095_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly Camarilla levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla: range = weekly_high - weekly_low
    weekly_range = weekly_high - weekly_low
    # Levels: close ± (range * multiplier)
    camarilla_h4 = weekly_close + weekly_range * 1.1/2
    camarilla_l4 = weekly_close - weekly_range * 1.1/2
    camarilla_h3 = weekly_close + weekly_range * 1.1/4
    camarilla_l3 = weekly_close - weekly_range * 1.1/4
    camarilla_h2 = weekly_close + weekly_range * 1.1/6
    camarilla_l2 = weekly_close - weekly_range * 1.1/6
    camarilla_h1 = weekly_close + weekly_range * 1.1/12
    camarilla_l1 = weekly_close - weekly_range * 1.1/12
    
    # Pivot bias: 1 if close > H3 (bullish), -1 if close < L3 (bearish), 0 otherwise (range)
    pivot_bias = np.where(weekly_close > camarilla_h3, 1,
                         np.where(weekly_close < camarilla_l3, -1, 0))
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1w, pivot_bias)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_bias_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Pivot Bias: from 1w data ---
        bullish_bias = pivot_bias_aligned[i] > 0
        bearish_bias = pivot_bias_aligned[i] < 0
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish weekly bias
            if breakout_up and bullish_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish weekly bias
            elif breakout_down and bearish_bias:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals