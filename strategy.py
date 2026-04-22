#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA filter and volume confirmation
# The Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trends when lines are aligned and separated.
# In trending markets: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend).
# In ranging markets: lines intertwine. We trade only when aligned (trending).
# 1d EMA(34) filter ensures we trade in direction of higher timeframe trend.
# Volume confirmation (>1.3x 20-period average) filters false signals.
# Designed for 6h timeframe targeting 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h data: Jaw(13), Teeth(8), Lips(5)
    # Smoothed with offset as per Williams: shift = (period+1)//2
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift((jaw_period+1)//2)
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift((teeth_period+1)//2)
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift((lips_period+1)//2)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_vals[i]) or 
            np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        lips_above_teeth = lips_vals[i] > teeth_vals[i]
        teeth_above_jaw = teeth_vals[i] > jaw_vals[i]
        lips_below_teeth = lips_vals[i] < teeth_vals[i]
        teeth_below_jaw = teeth_vals[i] < jaw_vals[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (uptrend) + price above 1d EMA + volume confirmation
            if (lips_above_teeth and teeth_above_jaw and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (downtrend) + price below 1d EMA + volume confirmation
            elif (lips_below_teeth and teeth_below_jaw and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (trend weakening) or price crosses 1d EMA
            if position == 1:
                # Exit long: Teeth < Lips (weakening uptrend) or price below 1d EMA
                if (teeth_vals[i] >= lips_vals[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Teeth > Lips (weakening downtrend) or price above 1d EMA
                if (teeth_vals[i] <= lips_vals[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0