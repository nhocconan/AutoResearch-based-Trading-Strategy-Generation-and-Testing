#!/usr/bin/env python3
"""
4h Williams Alligator Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trendless markets.
Price breaking above/below Alligator's Lips with Jaw-Teeth-Lips alignment,
volume spike, and 1d EMA34 trend captures momentum breakouts.
Works in bull/bear by trading both directions with trend filter.
Discrete sizing (0.0, ±0.30) targets 25-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h (primary timeframe)
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead  
    # Lips: 5-period SMMA smoothed 3 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().shift(3)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (13+8=21) and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Jaw > Teeth > Lips (uptrend) or Jaw < Teeth < Lips (downtrend)
        alligator_up = jaw_val > teeth_val > lips_val
        alligator_down = jaw_val < teeth_val < lips_val
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Lips AND Alligator aligned up AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_close > lips_val) and alligator_up and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Lips AND Alligator aligned down AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_close < lips_val) and alligator_down and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Lips OR Alligator alignment breaks OR price crosses below EMA
            if (curr_close < lips_val) or not alligator_up or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above Lips OR Alligator alignment breaks OR price crosses above EMA
            if (curr_close > lips_val) or not alligator_down or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Williams_Alligator_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0