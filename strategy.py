#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator's Jaw (TEMA13) and Lips > Teeth > Jaw alignment, 1w EMA21 rising, volume > 2x average
# Short when price < Alligator's Jaw and Jaws > Teeth > Lips alignment, 1w EMA21 falling, volume > 2x average
# Williams Alligator identifies trend phases; 1w EMA21 filters trend direction; volume confirms strength
# Targets 12-37 trades per year (50-150 over 4 years) for low fee drag and high win rate
# Works in bull markets via trend alignment and in bear markets via inverse alignment + volume spikes

name = "12h_WilliamsAlligator_1wEMA21_Volume"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (TEMA13), Teeth (TEMA8), Lips (TEMA5)
    # Using EMA as proxy for TEMA due to similar smoothing properties
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate EMA21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 2x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need at least 50 periods for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema21_1w_val = ema21_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price > Jaw, Lips > Teeth > Jaw alignment, 1w uptrend, volume confirmation
            if (close_val > jaw_val and lips_val > teeth_val and teeth_val > jaw_val and 
                ema21_1w_val > 0 and vol_conf_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw, Jaws > Teeth > Lips alignment, 1w downtrend, volume confirmation
            elif (close_val < jaw_val and jaw_val > teeth_val and teeth_val > lips_val and 
                  ema21_1w_val < 0 and vol_conf_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Jaw or alignment broken or 1w trend down
            if (close_val < jaw_val or lips_val <= teeth_val or teeth_val <= jaw_val or 
                ema21_1w_val <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Jaw or alignment broken or 1w trend up
            if (close_val > jaw_val or jaw_val <= teeth_val or teeth_val <= lips_val or 
                ema21_1w_val >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals