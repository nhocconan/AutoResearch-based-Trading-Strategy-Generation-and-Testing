#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator Jaw, Alligator Teeth > Lips, 1w EMA10 rising, volume > 1.5x average
# Short when price < Alligator Jaw, Alligator Teeth < Lips, 1w EMA10 falling, volume > 1.5x average
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Uses 12h for entry timing, 1w for trend filter to avoid whipsaws
# Targets 12-37 trades/year (50-150 total over 4 years) for low fee drag

name = "12h_WilliamsAlligator_1wTrend_Volume"
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
    
    # Get 12h data once for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h close
    close_12h = df_12h['close'].values
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # SMA(13)
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # SMA(8)
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # SMA(5)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate EMA10 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # warmup for Williams Alligator
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema10_1w_val = ema10_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price > jaw, teeth > lips, 1w uptrend, volume spike
            if close_val > jaw_val and teeth_val > lips_val and ema10_1w_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price < jaw, teeth < lips, 1w downtrend, volume spike
            elif close_val < jaw_val and teeth_val < lips_val and ema10_1w_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < jaw or teeth < lips or 1w trend down
            if close_val < jaw_val or teeth_val < lips_val or ema10_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > jaw or teeth > lips or 1w trend up
            if close_val > jaw_val or teeth_val > lips_val or ema10_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals