#!/usr/bin/env python3
"""
12h_Williams_Alligator_Trend_Filter_Volume_Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) on 12h with 1w trend filter and volume spike
provides robust trend-following signals. Uses Williams %R for entry timing on pullbacks.
Targets 20-40 trades/year. Works in bull via trend alignment, in bear via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period=14):
    """Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams Alligator and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    median_price = (df_1w['high'] + df_1w['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # 1w trend filter: price above/below 21-period EMA
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Williams %R on 12h for entry timing
    wr = williams_r(high, low, close, period=14)
    
    # Volume spike: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Alligator (13+8=21) and Williams %R (14)
    start_idx = max(21, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(wr[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema21_1w_aligned[i]
        wr_val = wr[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > EMA21 + Williams %R oversold + volume spike
            if (lips_val > teeth_val > jaw_val and 
                close[i] > ema_trend and 
                wr_val < -80 and 
                vol_spike_val):
                signals[i] = size
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < EMA21 + Williams %R overbought + volume spike
            elif (lips_val < teeth_val < jaw_val and 
                  close[i] < ema_trend and 
                  wr_val > -20 and 
                  vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish OR Williams %R overbought
            if (lips_val < teeth_val or wr_val > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator turns bullish OR Williams %R oversold
            if (lips_val > teeth_val or wr_val < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_Alligator_Trend_Filter_Volume_Spike"
timeframe = "12h"
leverage = 1.0