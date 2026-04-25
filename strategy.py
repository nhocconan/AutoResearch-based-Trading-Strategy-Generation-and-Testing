#!/usr/bin/env python3
"""
12h Williams Alligator Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence/presence.
Breakouts occur when LIPS crosses above/below TEETH/JAW with volume spike.
1w EMA50 ensures trading with the higher timeframe trend.
Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
12h timeframe targets 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator on 1d
    jaw, teeth, lips = compute_williams_alligator(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Alligator lines to 12h timeframe with extra delay (Alligator needs confirmation)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw, additional_delay_bars=1)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth, additional_delay_bars=1)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips, additional_delay_bars=1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # volume MA, 1w EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Alligator signals: LIPS crossing TEETH/JAW
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        lips_above_jaw = lips_aligned[i] > jaw_aligned[i]
        lips_below_jaw = lips_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: LIPS crosses above TEETH AND JAW (bullish alignment) AND uptrend AND volume spike
            long_entry = lips_above_teeth and lips_above_jaw and uptrend and vol_spike
            # Short: LIPS crosses below TEETH AND JAW (bearish alignment) AND downtrend AND volume spike
            short_entry = lips_below_teeth and lips_below_jaw and downtrend and vol_spike
            
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
            # Exit: LIPS crosses below TEETH (loss of bullish alignment) OR loss of uptrend
            if (lips_aligned[i] < teeth_aligned[i]) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: LIPS crosses above TEETH (loss of bearish alignment) OR loss of downtrend
            if (lips_aligned[i] > teeth_aligned[i]) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0