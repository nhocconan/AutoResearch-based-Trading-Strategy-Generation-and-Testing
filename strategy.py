#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
# Long when price > Alligator Jaw and price > 1w EMA50 with volume spike.
# Short when price < Alligator Jaw and price < 1w EMA50 with volume spike.
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trend direction.
# Williams Alligator works in both trending and ranging markets by identifying when the
# three lines are intertwined (sleeping) or separated (awake/hunting). Designed for 12h
# timeframe with weekly trend filter to reduce noise and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Target: 15-25 trades/year to minimize fee drag.
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
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w trend filter: 50-period EMA on close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator indicators
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    
    # Calculate SMMA (Smoothed Moving Average) using EMA as approximation
    # SMMA is similar to EMA but with different smoothing factor
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).values  # 13-period
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False).values   # 8-period
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False).values    # 5-period
    
    # Alligator Jaw is the middle line (Teeth in original, but we use median of three)
    # Using Teeth as the main trend indicator line
    jaw_aligned = jaw  # Already LTF aligned
    teeth_aligned = teeth
    lips_aligned = lips
    
    # 12h volume average for spike detection
    vol_ema_12h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_12h > 0, volume / vol_ema_12h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicator calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Alligator condition: lines are separated (awake) - Jaw > Teeth > Lips for uptrend, reverse for downtrend
        alligator_long = teeth_aligned[i] > lips_aligned[i] and jaw_aligned[i] > teeth_aligned[i]
        alligator_short = teeth_aligned[i] < lips_aligned[i] and jaw_aligned[i] < teeth_aligned[i]
        
        if position == 0:
            # Long condition: price above Jaw, in uptrend, Alligator awake (bullish alignment) with volume spike
            long_condition = (close[i] > jaw_aligned[i]) and uptrend and alligator_long and vol_spike[i]
            # Short condition: price below Jaw, in downtrend, Alligator awake (bearish alignment) with volume spike
            short_condition = (close[i] < jaw_aligned[i]) and downtrend and alligator_short and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Jaw or trend turns down or Alligator goes to sleep
            if (close[i] < jaw_aligned[i]) or (not uptrend) or (not alligator_long):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Jaw or trend turns up or Alligator goes to sleep
            if (close[i] > jaw_aligned[i]) or (not downtrend) or (not alligator_short):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals