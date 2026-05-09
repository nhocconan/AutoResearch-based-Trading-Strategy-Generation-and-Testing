#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Long when price is above 1w EMA50, Alligator lines are bullish (jaw < teeth < lips), and volume spike
# Short when price is below 1w EMA50, Alligator lines are bearish (jaw > teeth > lips), and volume spike
# Exit when price crosses back below/above teeth line or Alligator direction changes
# Uses Williams Alligator for trend/momentum, 1w EMA for higher timeframe trend filter, volume for confirmation
# Designed to work in trending markets via Alligator alignment and in ranging markets via line cross
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 13:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator (13, 8, 5 periods with future shifts)
    # Jaw (Blue line): 13-period SMMA shifted 8 bars forward
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean()
    jaw = (jaw + pd.Series(low).rolling(window=13, min_periods=13).mean()) / 2
    jaw = jaw.shift(8)  # Shift 8 bars forward
    
    # Teeth (Red line): 8-period SMMA shifted 5 bars forward
    teeth = pd.Series(high).rolling(window=8, min_periods=8).mean()
    teeth = (teeth + pd.Series(low).rolling(window=8, min_periods=8).mean()) / 2
    teeth = teeth.shift(5)  # Shift 5 bars forward
    
    # Lips (Green line): 5-period SMMA shifted 3 bars forward
    lips = pd.Series(high).rolling(window=5, min_periods=5).mean()
    lips = (lips + pd.Series(low).rolling(window=5, min_periods=5).mean()) / 2
    lips = lips.shift(3)  # Shift 3 bars forward
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need enough data for Jaw (13-period + 8 shift)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish Alligator (jaw < teeth < lips), price above 1w EMA50, volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                close[i] > ema50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator (jaw > teeth > lips), price below 1w EMA50, volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below teeth OR Alligator turns bearish (jaw > teeth)
            if (close[i] < teeth[i]) or (jaw[i] > teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above teeth OR Alligator turns bullish (jaw < teeth)
            if (close[i] > teeth[i]) or (jaw[i] < teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals