#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume spike.
# Long when Green line > Red line (bullish alignment) AND price > 1d EMA34 with volume spike.
# Short when Green line < Red line (bearish alignment) AND price < 1d EMA34 with volume spike.
# Uses Williams Alligator (Smoothed Medians) to identify trend direction and avoid chop.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 15-25/year) to reduce fee drag.
# Works in both bull and bear markets by following the 1d trend direction via EMA34 filter.
name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Three Smoothed Medians (Jaw, Teeth, Lips)
    # Jaw: 13-period Smoothed Median (8 offset)
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).median()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean().shift(8).values  # 13+8 offset
    
    # Teeth: 8-period Smoothed Median (5 offset)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).median()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean().shift(5).values  # 8+5 offset
    
    # Lips: 5-period Smoothed Median (3 offset)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).median()
    lips = lips_raw.rolling(window=3, min_periods=3).mean().shift(3).values  # 5+3 offset
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Alligator lines
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long condition: bullish alignment, in uptrend with volume spike
            long_condition = bullish_align and uptrend and vol_spike[i]
            # Short condition: bearish alignment, in downtrend with volume spike
            short_condition = bearish_align and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish alignment or trend turns down
            if bearish_align or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish alignment or trend turns up
            if bullish_align or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals