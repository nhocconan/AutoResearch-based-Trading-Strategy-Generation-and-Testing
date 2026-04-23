#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend strength and direction
- Only trade when Alligator is 'eating' (trending) in alignment with 1d EMA50
- Volume confirmation (> 1.8x 20-period average) filters false signals
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Alligator provides clear trend signals with fewer whipsaws than MA crossovers
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
    
    # Calculate Williams Alligator on 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    median_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    
    # Alligator lines: Jaw(13), Teeth(8), Lips(5) - all shifted forward
    jaw = pd.Series(median_4h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_4h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_4h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish (Lips > Teeth > Jaw) with 1d uptrend and volume spike
            alligator_bullish = (lips_aligned[i] > teeth_aligned[i] and 
                               teeth_aligned[i] > jaw_aligned[i])
            long_signal = (alligator_bullish and 
                          close[i] > ema_50_1d_aligned[i] and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: Alligator bearish (Lips < Teeth < Jaw) with 1d downtrend and volume spike
            alligator_bearish = (lips_aligned[i] < teeth_aligned[i] and 
                               teeth_aligned[i] < jaw_aligned[i])
            short_signal = (alligator_bearish and 
                           close[i] < ema_50_1d_aligned[i] and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator sleeping (no clear trend) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish or 1d trend turns bearish
                alligator_bearish = (lips_aligned[i] < teeth_aligned[i] and 
                                   teeth_aligned[i] < jaw_aligned[i])
                if alligator_bearish or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns bullish or 1d trend turns bullish
                alligator_bullish = (lips_aligned[i] > teeth_aligned[i] and 
                                   teeth_aligned[i] > jaw_aligned[i])
                if alligator_bullish or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0