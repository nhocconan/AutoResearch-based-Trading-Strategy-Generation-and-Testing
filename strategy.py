#!/usr/bin/env python3
"""
4h Williams Alligator Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Williams Alligator (JAWS/TEETH/LIPS) identifies trend absence/presence. 
Breakouts occur when Alligator lines converge (sleeping) then diverge (awakening) with price piercing 
the extreme line (JAWS for longs, LIPS for shorts) with volume confirmation and 1d EMA34 trend alignment.
Designed for low-moderate trade frequency (19-50/year) on 4h timeframe to work in both bull and bear 
markets by trading awakening trends with institutional participation confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h: SMAs of median price
    # Jaw (blue): 13-period SMMA shifted 8 bars
    # Teeth (red): 8-period SMMA shifted 5 bars
    # Lips (green): 5-period SMMA shifted 3 bars
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev_SMMA * (period-1) + Current_Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate shifted values
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator lines and volume MA
    start_idx = max(13, 20) + 8  # max period + jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator sleeping condition: lines intertwined (JAW > TEETH > LIPS for down, LIPS > TEETH > JAW for up)
        # Actually, sleeping is when they are close together; we'll use convergence
        # Simpler: trade when price breaks extreme line with alignment
        
        if position == 0:
            # Look for entry signals
            # Long: price crosses above JAW (upper line) AND Alligator awakening (JAW > TEETH > LIPS) 
            #        AND volume spike AND price > 1d EMA34 (uptrend)
            jaw_above_teeth = jaw_val > teeth_val
            teeth_above_lips = teeth_val > lips_val
            alligator_long = jaw_above_teeth and teeth_above_lips
            long_entry = (curr_close > jaw_val) and alligator_long and vol_spike and (curr_close > ema_trend)
            
            # Short: price crosses below LIPS (lower line) AND Alligator awakening (LIPS < TEETH < JAW)
            #        AND volume spike AND price < 1d EMA34 (downtrend)
            lips_below_teeth = lips_val < teeth_val
            teeth_below_jaw = teeth_val < jaw_val
            alligator_short = lips_below_teeth and teeth_below_jaw
            short_entry = (curr_close < lips_val) and alligator_short and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below TEETH (middle line) OR price crosses below EMA (trend change)
            if (curr_close < teeth_val) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above TEETH (middle line) OR price crosses above EMA (trend change)
            if (curr_close > teeth_val) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0