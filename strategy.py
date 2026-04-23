#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 Trend Filter and Volume Confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via SMAs with forward shift
- Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 + volume > 1.5x 20-period MA
- Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 + volume > 1.5x 20-period MA
- Exit when Alligator alignment breaks or price crosses 1d EMA50
- Uses 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) avoiding fee drag
- Works in bull/bear markets via 1d EMA50 trend filter and volume confirmation for validity
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 12h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Jaw (blue line): 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    # Teeth (red line): 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    # Lips (green line): 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish alignment + price > 1d EMA50 + volume spike
            if (bullish and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price < 1d EMA50 + volume spike
            elif (bearish and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment breaks OR price crosses 1d EMA50
            exit_signal = False
            bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            if position == 1:
                # Exit long when bearish alignment OR price < 1d EMA50
                if bearish or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when bullish alignment OR price > 1d EMA50
                if bullish or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0