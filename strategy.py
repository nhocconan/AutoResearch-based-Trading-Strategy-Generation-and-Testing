#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (JAW/TEETH/LIPS) with 1w EMA34 trend filter and volume confirmation.
Long when price > LIPS > TEETH > JAW (bullish alignment) AND price > 1w EMA34 (uptrend) AND volume > 1.8x average.
Short when price < LIPS < TEETH < JAW (bearish alignment) AND price < 1w EMA34 (downtrend) AND volume > 1.8x average.
Exit when Alligator alignment breaks (LIPS crosses TEETH or JAW) OR price crosses 1w EMA34.
Uses 12h timeframe with Alligator's smooth trend-following properties to reduce whipsaw.
1w EMA34 provides strong higher-timeframe trend filter. Volume spike ensures high-conviction entries.
Target: 60-120 trades over 4 years (15-30/year) to stay within proven working range for 12h.
Williams Alligator is effective in both bull and bear markets by identifying trending vs ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_12h = (df_12h['high'] + df_12h['low']) / 2.0
    median_12h = median_12h.values
    
    # Alligator lines: JAW (13,8), TEETH (8,5), LIPS (5,3)
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(80, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Bullish alignment: LIPS > TEETH > JAW
            bullish = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish alignment: LIPS < TEETH < JAW
            bearish = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long: bullish alignment AND price > 1w EMA34 (uptrend) AND volume spike
            if bullish and price > ema34_val and vol_current > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < 1w EMA34 (downtrend) AND volume spike
            elif bearish and price < ema34_val and vol_current > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Check if Alligator alignment is broken
            bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
            bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
            
            if position == 1:
                # Exit long: bullish alignment broken OR price < 1w EMA34 (trend reversal)
                if not bullish_aligned or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bearish alignment broken OR price > 1w EMA34 (trend reversal)
                if not bearish_aligned or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0