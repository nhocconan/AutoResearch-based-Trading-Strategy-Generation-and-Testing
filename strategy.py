#!/usr/bin/env python3
"""
12h_1w_Donchian20_1dVolumeBreakout_Regime_v1
Concept: 12h price breaks weekly Donchian(20) with daily volume spike and weekly chop filter.
- Long: Close > weekly Donchian Upper(20) AND daily volume > 2.0x 20-period avg AND WEEKLY CHOP(14) > 61.8 (range regime)
- Short: Close < weekly Donchian Lower(20) AND daily volume > 2.0x 20-period avg AND WEEKLY CHOP(14) > 61.8 (range regime)
- Exit: Close crosses back through weekly midline
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: weekly structure adapts, volume confirms institutional interest, chop filter avoids trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian20_1dVolumeBreakout_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly: Donchian Channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # === Daily: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Weekly: Chopiness Index (14) ===
    atr_period = 14
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of absolute returns
    returns = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    sum_returns = pd.Series(returns).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    chop = 100 * np.log10(sum_returns / (atr * atr_period)) / np.log10(atr_period)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === 12h: Price ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        mid_val = donchian_mid_aligned[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(mid_val) or 
            np.isnan(vol_ma_20) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 2.0x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 2.0 * vol_ma_20
        
        # Chop condition: range-bound market
        chop_condition = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper with volume spike and range regime
            if close[i] > upper_val and vol_condition and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower with volume spike and range regime
            elif close[i] < lower_val and vol_condition and chop_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly midline
            if close[i] < mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly midline
            if close[i] > mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals