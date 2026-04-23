#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator Jaw breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Alligator Jaw AND 1d EMA50 is rising AND volume > 1.8x 20-period average.
Short when price breaks below Alligator Jaw AND 1d EMA50 is falling AND volume > 1.8x 20-period average.
Exit when price touches the opposite side of the Alligator (Teeth for long, Lips for short) or reverses EMA50 direction.
Uses 1d HTF for EMA50 trend to reduce whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3). All calculated on median price.
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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate 1d Williams Alligator components (Jaw, Teeth, Lips) on median price
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_1d = (high_1d + low_1d) / 2.0
    
    # Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
    jaw_1d = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Alligator Jaw AND EMA50 rising AND volume spike
            if price > jaw and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Alligator Jaw AND EMA50 falling AND volume spike
            elif price < jaw and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Teeth OR EMA50 starts falling
                if price < teeth or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Lips OR EMA50 starts rising
                if price > lips or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_JawBreakout_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0