#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND close > 1d EMA50 (uptrend).
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND close < 1d EMA50 (downtrend).
Exit when price reverts to Camarilla pivot point (PP) OR EMA50 slope flips (trend change).
Uses 6h for Camarilla calculation and 1d for EMA50 filter to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise support/resistance,
volume confirmation filters fakeouts, EMA50 filter ensures alignment with daily trend.
Works in bull markets (captures uptrend continuations) and bear markets (captures downtrend continuations).
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
    
    # Get 6h data for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels on 6h timeframe (using previous period's OHLC)
    # Camarilla formulas: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    typical_price = (high_6h + low_6h + close_6h) / 3
    range_6h = high_6h - low_6h
    
    # Shift by 1 to use previous period's data (no look-ahead)
    pp = np.roll(typical_price, 1)
    r1 = np.roll(close_6h, 1) + np.roll(range_6h, 1) * 1.1 / 12
    s1 = np.roll(close_6h, 1) - np.roll(range_6h, 1) * 1.1 / 12
    
    # First value will be NaN due to roll, set to 0
    pp[0] = 0
    r1[0] = 0
    s1[0] = 0
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h Camarilla to 6h timeframe (no alignment needed)
    pp_aligned = pp
    r1_aligned = r1
    s1_aligned = s1
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume average (20-period) on 6h
    volume_6h = df_6h['volume'].values
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.3x avg AND close > 1d EMA50 (uptrend)
            if price > r1_val and vol > 1.3 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.3x avg AND close < 1d EMA50 (downtrend)
            elif price < s1_val and vol > 1.3 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla PP OR price < 1d EMA50 (trend change)
            if price < pp_val or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla PP OR price > 1d EMA50 (trend change)
            if price > pp_val or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Volume_EMA50_Filter"
timeframe = "6h"
leverage = 1.0