#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND volume > 1.5x average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND volume > 1.5x average.
Exit when Alligator alignment reverses or volume drops below average.
Uses 12h timeframe for lower trade frequency with strong trend signals to avoid fee drag.
1d Elder Ray provides trend strength filter. Volume confirmation ensures high-conviction moves.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Williams Alligator (SMMA with offsets)
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(close_12h, 13)
    teeth_raw = smma(close_12h, 8)
    lips_raw = smma(close_12h, 5)
    
    # Apply offsets as per Williams Alligator specification
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Load 1d data for Elder Ray - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment AND bull power > 0 AND volume spike
            if (jaw_val < teeth_val < lips_val and bull_val > 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND bear power < 0 AND volume spike
            elif (jaw_val > teeth_val > lips_val and bear_val < 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment breaks OR bull power <= 0
                if not (jaw_val < teeth_val < lips_val) or bull_val <= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment breaks OR bear power >= 0
                if not (jaw_val > teeth_val > lips_val) or bear_val >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dElderRay_Volume"
timeframe = "12h"
leverage = 1.0