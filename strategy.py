#!/usr/bin/env python3
"""
Experiment #4895: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) with volume confirmation (>1.5x average) capture strong momentum moves. Uses ATR(14) trailing stop (2.0x) to limit downside. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts above pivot with uptrend) and bear markets (breakdowns below pivot with downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4895_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data for pivot and trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Weekly Pivot (using prior week's data) ===
    if len(df_1w) >= 1:
        # Calculate weekly pivot from prior completed week
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Align to 6s timeframe (shifted by 1 week for completed bars only)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: ATR(14) for stoploss ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        # Align ATR to 6s timeframe
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    else:
        atr_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot alignment
        breakout_long = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
        
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