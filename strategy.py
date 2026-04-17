#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, price > EMA50 (1d), and volume > 1.5x average.
Short when Bull Power < 0, Bear Power > 0, price < EMA50 (1d), and volume > 1.5x average.
Exit when Elder Ray signals weaken (Bull Power <= 0 for long, Bear Power <= 0 for short) OR volume drops.
Uses 6h for Elder Ray calculation and 1d for EMA50 filter to reduce whipsaw and align with higher timeframe trend.
Targets 50-150 total trades over 4 years (12-37/year). Elder Ray measures bull/bear power via EMA,
volume confirms conviction, and 1d EMA50 ensures trading with the dominant trend.
Works in bull markets (captures sustained buying pressure) and bear markets (captures sustained selling pressure).
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
    
    # Get 6h data for Elder Ray and EMA13 (fast EMA for Bull/Bear Power)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA13 on 6h for Elder Ray (Bull/Bear Power)
    close_6h_series = pd.Series(close_6h)
    ema13 = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13
    # Bear Power = Low - EMA13
    bear_power = low_6h - ema13
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h indicators to 6h timeframe (no alignment needed)
    bull_power_aligned = bull_power
    bear_power_aligned = bear_power
    
    # Align 1d EMA50 to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power_aligned[i]
        br = bear_power_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > EMA50 AND volume > 1.5x avg
            if bp > 0 and br < 0 and price > ema50_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND price < EMA50 AND volume > 1.5x avg
            elif bp < 0 and br > 0 and price < ema50_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 (weakening bullish pressure) OR volume < average
            if bp <= 0 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= 0 (weakening bearish pressure) OR volume < average
            if br <= 0 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Volume_EMA50_Filter"
timeframe = "6h"
leverage = 1.0