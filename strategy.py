#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Adaptive
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend regime filter and ATR-based stops.
Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 (bull regime).
Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA50 (bear regime).
ATR(10) trailing stop (2.5x) and Camarilla R4/S4 as profit targets.
Designed for both bull and bear markets: regime filter adapts to 1d trend, Elder Ray measures underlying power.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA50 trend regime and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend regime
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(10) on 6h for trailing stop and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Elder Ray components on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Camarilla R4 and S4 from prior 1d bar (for profit targets)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    if len(high_1d) < 2:
        camarilla_r4 = np.full_like(close_1d_arr, np.nan)
        camarilla_s4 = np.full_like(close_1d_arr, np.nan)
    else:
        camarilla_r4 = close_1d_arr[:-1] + 1.5 * (high_1d[:-1] - low_1d[:-1])  # R4 = close + 1.5*(range)
        camarilla_s4 = close_1d_arr[:-1] - 1.5 * (high_1d[:-1] - low_1d[:-1])  # S4 = close - 1.5*(range)
        camarilla_r4 = np.concatenate([[np.nan], camarilla_r4])
        camarilla_s4 = np.concatenate([[np.nan], camarilla_s4])
    
    # Align 1d indicators to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA13 (13), ATR (10), 1d EMA50 (50)
    start_idx = max(13, 10, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying strength), Bear Power < 0 (no selling pressure), price > 1d EMA50 (bull regime)
            long_signal = (bull_val > 0) and (bear_val < 0) and (close_val > ema_50_1d_val)
            # Short: Bear Power > 0 (selling pressure), Bull Power < 0 (no buying strength), price < 1d EMA50 (bear regime)
            short_signal = (bear_val > 0) and (bull_val < 0) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price reaches Camarilla R4 (profit target)
            elif close_val >= r4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: regime change (price < 1d EMA50)
            elif close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price reaches Camarilla S4 (profit target)
            elif close_val <= s4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: regime change (price > 1d EMA50)
            elif close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "6h_ElderRay_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0