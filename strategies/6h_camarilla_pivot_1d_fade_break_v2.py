#!/usr/bin/env python3
"""
6h Camarilla pivot with volume and ATR filter
Hypothesis: Camarilla pivot levels on 1d provide institutional support/resistance. 
Fade at R3/S3 with volume confirmation, breakout continuation at R4/S4 with volume surge.
Works in both bull (fade resistance, break support) and bear (fade support, break resistance).
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_fade_break_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas (using previous day's range)
    range_1d = high_1d - low_1d
    close_prev = close_1d  # using same day's close for same-day levels (will be shifted by align)
    
    # Calculate levels
    r4 = close_prev + (range_1d * 1.500)
    r3 = close_prev + (range_1d * 1.250)
    r2 = close_prev + (range_1d * 1.166)
    r1 = close_prev + (range_1d * 1.083)
    s1 = close_prev - (range_1d * 1.083)
    s2 = close_prev - (range_1d * 1.166)
    s3 = close_prev - (range_1d * 1.250)
    s4 = close_prev - (range_1d * 1.500)
    
    # Align levels to 6h timeframe (shifted by 1 day to avoid look-ahead)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    vol_filter = volume > (1.5 * vol_ema)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or reversal at S3
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] <= s3_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or reversal at R3
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] >= r3_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Fade at S3/R3 with volume confirmation
            fade_long = (close[i] <= s3_6h[i] * 1.002 and  # slightly above S3
                        volume[i] > vol_ema[i] * 1.3)
            fade_short = (close[i] >= r3_6h[i] * 0.998 and  # slightly below R3
                         volume[i] > vol_ema[i] * 1.3)
            
            # Breakout at S4/R4 with volume surge
            breakout_long = (close[i] > s4_6h[i] and
                           volume[i] > vol_ema[i] * 2.0)
            breakout_short = (close[i] < r4_6h[i] and
                            volume[i] > vol_ema[i] * 2.0)
            
            if fade_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif fade_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals