#!/usr/bin/env python3
"""
Experiment #035: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot direction (price > weekly pivot = long bias, < = short bias) and confirmed by 1d volume spike (>2x average), captures high-probability trend continuation moves. Weekly pivot provides structural bias from higher timeframe, Donchian breaks capture momentum, and volume filter ensures institutional participation. Designed for low turnover (target: 12-37 trades/year) to minimize fee drag while working in both bull (breakout longs) and bear (breakdown shorts) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    else:
        pivot_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate Donchian(20) channels
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Use pandas rolling for efficiency
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
        donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_period)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Directional Filter: Weekly pivot bias ---
        # Price above weekly pivot = long bias, below = short bias
        long_bias = close[i] > pivot_1w_aligned[i]
        short_bias = close[i] < pivot_1w_aligned[i]
        
        # --- Volume Confirmation: Require significant volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian low break (trailing stop) or opposite Donchian break
                if close[i] <= donchian_low[i] or close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian high break (trailing stop) or opposite Donchian break
                if close[i] >= donchian_high[i] or close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian high breakout with volume spike and long bias
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above Donchian high
            volume_spike and                 # Volume confirmation
            long_bias                        # Weekly pivot bias long
        )
        
        # Short: Donchian low breakdown with volume spike and short bias
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below Donchian low
            volume_spike and                 # Volume confirmation
            short_bias                       # Weekly pivot bias short
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals