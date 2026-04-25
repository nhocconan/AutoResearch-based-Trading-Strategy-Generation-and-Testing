#!/usr/bin/env python3
"""
12h Williams Alligator Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence on 12h.
When Alligator lines are intertwined (sleeping), we wait for breakout with volume spike.
Trend filter from 1d EMA34 ensures we trade in direction of higher timeframe momentum.
Works in bull via breakouts above lips, in bear via breakdowns below jaws.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-30 trades/year on 12h.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 12h data for Williams Alligator (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h: SMA of median price (hlc3)
    # Jaw: 13-period SMA, 8 bars ahead
    # Teeth: 8-period SMA, 5 bars ahead  
    # Lips: 5-period SMA, 3 bars ahead
    hlc3 = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe (already on 12h, so direct use)
    jaw_aligned = jaw  # Already aligned to 12h
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator sleeping condition: lines intertwined (market ranging)
        # We trade breakouts from this condition
        alligator_sleeping = (
            (abs(jaw_val - teeth_val) < (jaw_val * 0.001)) and  # Jaw-Teeth close
            (abs(teeth_val - lips_val) < (teeth_val * 0.001)) and  # Teeth-Lips close
            (abs(lips_val - jaw_val) < (lips_val * 0.001))   # Lips-Jaw close
        )
        
        if position == 0:
            # Look for entry signals only when Alligator is sleeping (ranging market)
            if alligator_sleeping:
                # Long: price breaks above Lips AND volume spike AND price > EMA34 (uptrend)
                long_entry = (curr_close > lips_val) and vol_spike and (curr_close > ema_trend)
                # Short: price breaks below Jaw AND volume spike AND price < EMA34 (downtrend)
                short_entry = (curr_close < jaw_val) and vol_spike and (curr_close < ema_trend)
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Wait for Alligator to sleep
        elif position == 1:
            # Long position management
            # Exit: price crosses below Teeth OR price crosses below EMA34
            if (curr_close < teeth_val) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Teeth OR price crosses above EMA34
            if (curr_close > teeth_val) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0