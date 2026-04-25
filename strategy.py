#!/usr/bin/env python3
"""
12h Williams Alligator Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Williams Alligator identifies trend presence and direction. Breakouts above the Alligator's teeth (middle line) 
with volume confirmation and aligned weekly EMA34 trend capture strong continuation moves. The weekly EMA34 ensures we 
trade with higher timeframe momentum, reducing false breakouts. Volume spike confirms participation. Designed for low 
trade frequency (12-37/year) on 12h timeframe to work in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend and Alligator (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on weekly close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on weekly data
    jaw, teeth, lips = compute_williams_alligator(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, Alligator, volume MA
    start_idx = max(34, 20) + 10  # extra buffer for indicator alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Alligator teeth AND volume spike AND price > weekly EMA34 (uptrend)
            # Alligator must be aligned (jaws < teeth < lips for uptrend)
            long_entry = (curr_close > teeth_val) and vol_spike and (curr_close > ema_trend) and (jaw_val < teeth_val < lips_val)
            # Short: price breaks below Alligator teeth AND volume spike AND price < weekly EMA34 (downtrend)
            # Alligator must be aligned (jaws > teeth > lips for downtrend)
            short_entry = (curr_close < teeth_val) and vol_spike and (curr_close < ema_trend) and (jaw_val > teeth_val > lips_val)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Alligator jaws OR price crosses below weekly EMA (trend change)
            if (curr_close < jaw_val) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Alligator lips OR price crosses above weekly EMA (trend change)
            if (curr_close > lips_val) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0