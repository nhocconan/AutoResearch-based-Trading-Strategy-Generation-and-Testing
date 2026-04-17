#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + volume confirmation + ATR trailing stop.
Long when Alligator jaws (13-period SMA shifted 8) < teeth (8-period SMA shifted 5) < lips (5-period SMA shifted 3) AND volume > 1.3x average.
Short when jaws > teeth > lips AND volume > 1.3x average.
Exit when price touches 8-period ATR-based trailing stop from extreme.
Alligator identifies trend direction with built-in smoothing, volume confirms momentum,
ATR stop manages risk in volatile crypto markets. Works in bull (captures uptrends) and bear (captures downtrends).
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 4h data for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Williams Alligator on 4h timeframe
    close_4h_series = pd.Series(close_4h)
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = close_4h_series.rolling(window=5, min_periods=5).mean().shift(3).values
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = close_4h_series.rolling(window=8, min_periods=8).mean().shift(5).values
    # Jaws: 13-period SMA, shifted 8 bars forward
    jaws = close_4h_series.rolling(window=13, min_periods=13).mean().shift(8).values
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # ATR (14-period) for trailing stop
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    close_4h_series = pd.Series(close_4h)
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Align Alligator lines to 4h timeframe (no alignment needed for same TF)
    lips_aligned = lips
    teeth_aligned = teeth
    jaws_aligned = jaws
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    # For trailing stop: track extreme price since entry
    long_extreme = 0.0
    short_extreme = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaws_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaws_val = jaws_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        atr_val = atr_aligned[i]
        
        if position == 0:
            # Long: lips < teeth < jaws (Alligator eating up, uptrend) AND volume > 1.3x avg
            if lips_val < teeth_val and teeth_val < jaws_val and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
                long_extreme = price  # initialize extreme
            # Short: lips > teeth > jaws (Alligator eating down, downtrend) AND volume > 1.3x avg
            elif lips_val > teeth_val and teeth_val > jaws_val and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
                short_extreme = price  # initialize extreme
        
        elif position == 1:
            # Update long extreme
            if price > long_extreme:
                long_extreme = price
            # Calculate trailing stop: long_extreme - 2.0 * ATR
            trailing_stop = long_extreme - 2.0 * atr_val
            # Exit long: price <= trailing stop
            if price <= trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update short extreme (lowest price since entry)
            if price < short_extreme:
                short_extreme = price
            # Calculate trailing stop: short_extreme + 2.0 * ATR
            trailing_stop = short_extreme + 2.0 * atr_val
            # Exit short: price >= trailing stop
            if price >= trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0