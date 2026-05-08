#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator jaws, 1w EMA50 rising, volume > 1.5x average
# Short when price < Alligator jaws, 1w EMA50 falling, volume > 1.5x average
# Uses Williams Alligator for trend identification, 1w EMA50 for higher timeframe trend filter
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag and high win rate

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    median_price = (high_12h + low_12h) / 2
    
    # Alligator lines: Jaw (13-period smoothed 8 bars ahead), Teeth (8-period smoothed 5 bars ahead), Lips (5-period smoothed 3 bars ahead)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift to avoid look-ahead (Williams Alligator uses future smoothing)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need at least 13 periods for Alligator jaw
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        lips_val = lips_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price > Alligator jaws, 1w uptrend, volume confirmation
            if close_val > jaw_val and close_val > lips_val and ema50_1w_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Alligator jaws, 1w downtrend, volume confirmation
            elif close_val < jaw_val and close_val < lips_val and ema50_1w_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Alligator lips or 1w trend down
            if close_val < lips_val or ema50_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Alligator lips or 1w trend up
            if close_val > lips_val or ema50_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals