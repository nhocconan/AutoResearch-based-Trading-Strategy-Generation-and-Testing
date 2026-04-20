#!/usr/bin/env python3
"""
12h_1w_1d_Adaptive_Breakout_With_Pullback_v1
Concept: Adaptive breakout strategy using weekly pivot points for trend, daily Donchian channels for entry timing,
         and volume confirmation to avoid false breakouts. Designed to work in both bull and bear markets.
- Trend: Use weekly pivot points - price above weekly PP = bullish bias, below = bearish bias
- Entry: In bullish bias, buy when price breaks above daily Donchian upper (20) with volume confirmation
         In bearish bias, sell when price breaks below daily Donchian lower (20) with volume confirmation
- Exit: Exit when price returns to the weekly pivot point (mean reversion to equilibrium)
- Volume: Require volume > 1.5x 20-period average for breakout confirmation
- Position sizing: 0.25 to manage risk
- Timeframe: 12h (primary), HTF: 1w (trend), 1d (entry/exit levels)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Adaptive_Breakout_With_Pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for entry/exit levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Weekly: Pivot Points for Trend Bias ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*PP - L
    r1_1w = 2 * pp_1w - low_1w
    # S1 = 2*PP - H
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivot levels to 12h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Daily: Donchian Channel for Entry/Exit ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === 12h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average for breakout confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        upper_20 = high_20_aligned[i]
        lower_20 = low_20_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(upper_20) or np.isnan(lower_20) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        # Determine bias: above weekly PP = bullish, below = bearish
        is_bullish_bias = current_close > pp_val
        is_bearish_bias = current_close < pp_val
        
        if position == 0:
            # Long entry: bullish bias + price breaks above Donchian high + volume
            if is_bullish_bias and current_close > upper_20 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish bias + price breaks below Donchian low + volume
            elif is_bearish_bias and current_close < lower_20 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot (mean reversion)
            if current_close < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot (mean reversion)
            if current_close > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals