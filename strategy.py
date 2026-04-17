#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
Long when Bull Power > 0, price > 1d EMA50, and volume > 1.5x average.
Short when Bear Power < 0, price < 1d EMA50, and volume > 1.5x average.
Exit when power reverses or volume drops.
Elder Ray measures bull/bear strength relative to EMA; combines trend (EMA) and momentum (power).
Works in bull markets (captures strong uptrends via Bull Power) and bear markets (captures downtrends via Bear Power).
Target: 50-150 total trades over 4 years (12-37/year). Discreet sizing to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA13 on 6h for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema13 = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_6h - ema13
    bear_power = low_6h - ema13
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        ema50 = ema50_1d_aligned[i]
        vol_ma = volume_ma_6h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0, price > 1d EMA50, volume > 1.5x average
            if bp > 0 and price > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, price < 1d EMA50, volume > 1.5x average
            elif br < 0 and price < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR price < 1d EMA50
            if bp <= 0 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 OR price > 1d EMA50
            if br >= 0 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Volume_EMA50_Filter"
timeframe = "6h"
leverage = 1.0