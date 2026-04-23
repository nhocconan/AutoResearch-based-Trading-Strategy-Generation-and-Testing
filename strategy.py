#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA34 Trend + Volume Spike
- Williams Alligator (JAWS=13, TEETH=8, LIPS=5) identifies trend direction and strength
- Alligator is "sleeping" (lines intertwined) in ranging markets, "awakening" (lines diverging) in trends
- Only trade when Alligator is awakening AND aligned with 1d EMA34 trend
- Volume confirmation (> 1.8x 24-period MA) filters false signals
- Designed for 12h timeframe to capture medium-term trends with controlled frequency (target: 12-37 trades/year)
- Uses smoothed median price (typical price) for Alligator calculation
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
    
    # Typical price for Alligator calculation
    typical_price = (high + low + close) / 3.0
    
    # Williams Alligator lines (smoothed with SMMA - using EMA as approximation)
    # JAWS: 13-period SMMA, shifted 8 bars forward
    # TEETH: 8-period SMMA, shifted 5 bars forward  
    # LIPS: 5-period SMMA, shifted 3 bars forward
    jaws = pd.Series(typical_price).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(typical_price).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(typical_price).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 24)  # need Alligator jaws, 1d EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator awakening: JAWS > TEETH > LIPS (uptrend) or JAWS < TEETH < LIPS (downtrend)
            jaws_gt_teeth = jaws[i] > teeth[i]
            teeth_gt_lips = teeth[i] > lips[i]
            jaws_lt_teeth = jaws[i] < teeth[i]
            teeth_lt_lips = teeth[i] < lips[i]
            
            # Long: Alligator awakening uptrend AND price > 1d EMA34 AND volume spike
            if (jaws_gt_teeth and teeth_gt_lips and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator awakening downtrend AND price < 1d EMA34 AND volume spike
            elif (jaws_lt_teeth and teeth_lt_lips and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator starts sleeping (lines intertwine) OR price crosses 1d EMA34
            exit_signal = False
            
            # Check if Alligator is sleeping (lines intertwining)
            jaws_teeth_close = abs(jaws[i] - teeth[i]) < (abs(jaws[i]) * 0.001)  # 0.1% threshold
            teeth_lips_close = abs(teeth[i] - lips[i]) < (abs(teeth[i]) * 0.001)  # 0.1% threshold
            alligator_sleeping = jaws_teeth_close and teeth_lips_close
            
            if position == 1:
                # Exit long when Alligator sleeping OR price < 1d EMA34
                if alligator_sleeping or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator sleeping OR price > 1d EMA34
                if alligator_sleeping or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0