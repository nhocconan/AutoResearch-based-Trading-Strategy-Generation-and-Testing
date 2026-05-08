#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsAlligator_Trend_Filter"
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
    
    # Get 1d data once for Alligator components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator components (13,8,5 periods with 8,5,3 offsets)
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high_series + low_series) / 2
    jaw_raw = median_price.rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = median_price.rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = median_price.rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Align to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Williams %R (14-period)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - df_1d['close']) / (highest_high - lowest_low))
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r.values)
    
    # Volume spike detection on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(williams_r_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        jaw_val = jaw_6h[i]
        teeth_val = teeth_6h[i]
        lips_val = lips_6h[i]
        
        # Williams %R extremes: oversold < -80, overbought > -20
        wr = williams_r_6h[i]
        
        if position == 0:
            # Long entry: Alligator aligned up + Williams %R oversold + volume spike
            long_cond = (jaw_val > teeth_val > lips_val and wr < -80 and vol_spike[i])
            
            # Short entry: Alligator aligned down + Williams %R overbought + volume spike
            short_cond = (jaw_val < teeth_val < lips_val and wr > -20 and vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator reverses down OR Williams %R overbought
            if (jaw_val < teeth_val or wr > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reverses up OR Williams %R oversold
            if (jaw_val > teeth_val or wr < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator + Williams %R on 1d timeframe with volume spike confirmation on 6h.
# The Alligator (Jaw/Teeth/Lips) identifies trend direction on higher timeframe.
# Williams %R identifies overbought/oversold conditions for entry timing.
# Volume spike confirms institutional participation.
# Long: Alligator bullish (Jaw>Teeth>Lips) + Williams %R < -80 (oversold) + volume spike.
# Short: Alligator bearish (Jaw<Teeth<Lips) + Williams %R > -20 (overbought) + volume spike.
# Exits when trend weakens or momentum reverses.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Targets 15-25 trades/year on 6h timeframe. Uses discrete sizing (0.25) to minimize churn.