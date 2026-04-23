#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA34 Trend Filter and Volume Confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength
- Enter long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA34 + volume > 1.5x 20-period MA
- Enter short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA34 + volume > 1.5x 20-period MA
- Exit when Alligator alignment breaks or price crosses 1d EMA34
- Uses 12h timeframe for lower trade frequency (target: 12-37 trades/year) to minimize fee drag
- Works in both bull and bear markets via 1d EMA34 trend filter and volume confirmation
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Median price for Alligator calculation
    median_price = (high + low) / 2
    
    # Calculate smoothed medians (SMMA - smoothed moving average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    smma_jaw = smma(median_price, jaw_period)
    smma_teeth = smma(median_price, teeth_period)
    smma_lips = smma(median_price, lips_period)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.roll(smma_jaw, jaw_shift)
    teeth = np.roll(smma_teeth, teeth_shift)
    lips = np.roll(smma_lips, lips_shift)
    
    # Invalidate shifted values
    jaw[:jaw_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    # Align Alligator lines to 12h timeframe (data is already 12h)
    jaw_aligned = jaw  # Already on 12h timeframe
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, jaw_shift, teeth_shift, lips_shift)  # need EMA34_1d, vol MA, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 (uptrend) AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment breaks OR price crosses 1d EMA34
            exit_signal = False
            if position == 1:
                # Exit long when bullish alignment breaks OR price < 1d EMA34
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when bearish alignment breaks OR price > 1d EMA34
                if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0