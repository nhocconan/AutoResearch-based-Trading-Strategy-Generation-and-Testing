#!/usr/bin/env python3
"""
1d Williams Alligator + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trend absence/presence.
When all three lines are entwined (no trend), market is ranging. When they diverge
(Jaw < Teeth < Lips for uptrend, reverse for downtrend), trend is strong.
We trade in direction of 1w EMA50 trend only when Alligator shows alignment,
avoiding choppy markets. Volume spike confirms institutional participation.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
1d timeframe targets 7-25 trades/year (30-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is SMA
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV SMMA * (period-1) + CURRENT) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    median_price = (high + low) / 2.0  # Williams Alligator uses median price
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator using SMMA (Smoothed Moving Average)
    # Jaw: SMMA(median, 13) shifted 8 bars
    # Teeth: SMMA(median, 8) shifted 5 bars  
    # Lips: SMMA(median, 5) shifted 3 bars
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # First shifted values are invalid
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13, 8, 5, 50)  # volume MA, Alligator periods, 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend_1w = curr_close > ema_50_1w_aligned[i]
        downtrend_1w = curr_close < ema_50_1w_aligned[i]
        
        # Alligator alignment: 
        # Uptrend: Lips > Teeth > Jaw (green alignment, all diverging upward)
        # Downtrend: Lips < Teeth < Jaw (red alignment, all diverging downward)
        alligator_uptrend = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
        alligator_downtrend = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator uptrend AND 1w uptrend AND volume spike
            long_entry = alligator_uptrend and uptrend_1w and vol_spike
            # Short: Alligator downtrend AND 1w downtrend AND volume spike
            short_entry = alligator_downtrend and downtrend_1w and vol_spike
            
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
            # Exit: Alligator loses uptrend alignment OR 1w trend turns down
            if not alligator_uptrend or downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses downtrend alignment OR 1w trend turns up
            if not alligator_downtrend or uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0