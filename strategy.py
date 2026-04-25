#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend phases on 6h; when Lips cross above Teeth (bullish) or below Teeth (bearish) with 1d EMA50 trend alignment and volume confirmation, it captures sustainable momentum. Designed for 6h to target 12-37 trades/year (50-150 over 4 years), minimizing fee drag. Works in both bull and bear markets by following 1d trend and avoiding counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 6h: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)    # 5-period, shifted 3
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(13+8, 8+5, 5+3, 50, 20)  # Alligator shifts + EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator signals: Lips cross Teeth
        lips_above_teeth = lips[i] > teeth[i]
        lips_below_teeth = lips[i] < teeth[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator cross + trend + volume
            # Long: Lips cross above Teeth AND bullish bias AND volume spike
            long_entry = lips_above_teeth and bullish_bias and vol_spike
            # Short: Lips cross below Teeth AND bearish bias AND volume spike
            short_entry = lips_below_teeth and bearish_bias and vol_spike
            
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
            # Exit: Lips cross back below Teeth (trend change) OR loss of bullish bias
            if lips_below_teeth or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Lips cross back above Teeth (trend change) OR loss of bearish bias
            if lips_above_teeth or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0