#!/usr/bin/env python3
"""
Hypothesis: 4h 1d Camarilla R1/S1 breakout with volume confirmation and ATR stop.
Long when price breaks above 1d Camarilla R1 level AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S1 level AND volume > 1.5x 20-period average.
Exit when price retraces 30% of ATR from extreme favorable price since entry.
Uses proven Camarilla pivot structure with volume filter to reduce false breakouts.
Designed for low trade frequency (20-50/year) to minimize fee drag while capturing strong intraday moves.
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
    
    # Get 1d data for Camarilla calculation (HTF timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rng_1d = high_1d - low_1d
    camarilla_r1_1d = close_1d + rng_1d * 1.1 / 12
    camarilla_s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Calculate volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) on 1d for stop loss
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First bar: use high-low
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    extreme_price = 0.0  # Tracks best price since entry for trailing stop
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        atr = atr_aligned[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 1.5x avg
            if high_price > r1 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                extreme_price = price
            # Short: price breaks below S1 AND volume > 1.5x avg
            elif low_price < s1 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                extreme_price = price
        
        elif position == 1:
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Exit long: price retraces 30% of ATR from extreme price
            if price < extreme_price - 0.3 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update extreme price (lowest since entry)
            if price < extreme_price:
                extreme_price = price
            # Exit short: price retraces 30% of ATR from extreme price
            if price > extreme_price + 0.3 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R1S1_Volume_ATRTrail"
timeframe = "4h"
leverage = 1.0