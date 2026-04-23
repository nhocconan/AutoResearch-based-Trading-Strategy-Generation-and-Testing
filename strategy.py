#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
Long when Jaw < Teeth < Lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x average.
Short when Jaw > Teeth > Lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x average.
Exit when Alligator alignment reverses or volume drops below average.
Williams Alligator identifies trend phases via smoothed medians (Jaw=13, Teeth=8, Lips=5).
1d EMA50 ensures trading in direction of higher timeframe trend.
Volume confirmation avoids low-conviction breakouts.
Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking trades aligned with 1d trend.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator on 12h data (SMMA of median price)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw).rolling(window=8, min_periods=8).mean().values  # SMMA smoothing
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean().values  # SMMA smoothing
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean().values  # SMMA smoothing
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Alligator alignment conditions
        bullish_aligned = (jaw_val < teeth_val) and (teeth_val < lips_val)
        bearish_aligned = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        if position == 0:
            # Long: Bullish alignment AND price > 1d EMA50 AND volume spike
            if bullish_aligned and (price > ema50_val) and (vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < 1d EMA50 AND volume spike
            elif bearish_aligned and (price < ema50_val) and (vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment turns bearish OR volume drops below average
                if not bullish_aligned or (vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment turns bullish OR volume drops below average
                if not bearish_aligned or (vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0