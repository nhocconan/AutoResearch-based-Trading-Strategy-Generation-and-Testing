#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike
# Long when Jaw > Teeth > Lips (bullish alignment) and price > Lips with volume > 2x average
# Short when Jaw < Teeth < Lips (bearish alignment) and price < Lips with volume > 2x average
# Exit when Alligator lines cross in opposite direction or price crosses Jaw
# Uses Williams Alligator for trend identification, EMA for higher timeframe trend, volume for conviction
# Designed to capture strong trends while avoiding choppy markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Williams_Alligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h data (Jaw=13, Teeth=8, Lips=5 SMAs with future shifts)
    # We calculate on close prices
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8)   # 13-period SMA shifted 8 bars forward
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5)    # 8-period SMA shifted 5 bars forward
    lips = close_series.rolling(window=5, min_periods=5).mean().shift(3)     # 5-period SMA shifted 3 bars forward
    
    jaw_arr = jaw.values
    teeth_arr = teeth.values
    lips_arr = lips.values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_arr[i]) or np.isnan(teeth_arr[i]) or np.isnan(lips_arr[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bullish alignment (Jaw > Teeth > Lips), price > Lips, volume spike
            if (jaw_arr[i] > teeth_arr[i] and teeth_arr[i] > lips_arr[i] and
                close[i] > lips_arr[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish alignment (Jaw < Teeth < Lips), price < Lips, volume spike
            elif (jaw_arr[i] < teeth_arr[i] and teeth_arr[i] < lips_arr[i] and
                  close[i] < lips_arr[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish alignment OR price crosses below Jaw
            if (jaw_arr[i] < teeth_arr[i] and teeth_arr[i] < lips_arr[i]) or (close[i] < jaw_arr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish alignment OR price crosses above Jaw
            if (jaw_arr[i] > teeth_arr[i] and teeth_arr[i] > lips_arr[i]) or (close[i] > jaw_arr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals