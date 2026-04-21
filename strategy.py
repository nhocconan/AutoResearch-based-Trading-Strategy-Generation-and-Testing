#!/usr/bin/env python3
"""
6h_1d_Pivot_R2S2_MomentumBreakout
Hypothesis: 6h timeframe with 1d pivot point levels - buy breakouts above R2 with momentum (price > 20 EMA) and volume confirmation, sell breakdowns below S2 with momentum (price < 20 EMA) and volume confirmation. Uses pivot points from daily data for key support/resistance levels. Designed to capture strong momentum moves in both bull and bear markets by trading breakouts of significant daily levels with momentum and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point and support/resistance levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R2 = PP + (High - Low)
    # S2 = PP - (High - Low)
    pp = (prev_high + prev_low + prev_close) / 3
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    
    # Align to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # EMA20 for momentum filter
    close_series = prices['close']
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = prices['volume']
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter
        volume_ok = volume > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long conditions: break above R2 + price > EMA20 (bullish momentum) + volume
            if price > r2_aligned[i] and price > ema_20[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S2 + price < EMA20 (bearish momentum) + volume
            elif price < s2_aligned[i] and price < ema_20[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R2S2_MomentumBreakout"
timeframe = "6h"
leverage = 1.0