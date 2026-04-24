#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume spike confirmation.
- Uses Williams Alligator from 1d timeframe: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs.
- Long when Lips > Teeth > Jaw (bullish alignment) AND price breaks above Teeth with volume > 1.8x 20-bar average.
- Short when Lips < Teeth < Jaw (bearish alignment) AND price breaks below Teeth with volume > 1.8x 20-bar average.
- Trend filter: price must be above/below 1d EMA50 to align with daily trend.
- Designed for 4h timeframe to capture medium-term trends with Alligator's trend-following strength.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (80-200 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false signals in choppy markets.
- Novelty: Combines Williams Alligator's trend alignment with 1d EMA filter and volume confirmation on 4h.
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
    
    # Get 1d data ONCE before loop for Alligator and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator components from 1d timeframe
    close_1d = df_1d['close'].values
    
    # Jaw: 13-period SMMA, smoothed by 8 periods
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMMA, smoothed by 5 periods
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMMA, smoothed by 3 periods
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Bullish Alligator alignment: Lips > Teeth > Jaw
                bullish_align = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
                # Bearish Alligator alignment: Lips < Teeth < Jaw
                bearish_align = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
                
                # Long: bullish alignment AND price breaks above Teeth AND above 1d EMA50
                if bullish_align and close[i] > teeth_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish alignment AND price breaks below Teeth AND below 1d EMA50
                elif bearish_align and close[i] < teeth_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR price closes below Teeth OR below 1d EMA50
            if (lips_aligned[i] < teeth_aligned[i] or 
                close[i] < teeth_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR price closes above Teeth OR above 1d EMA50
            if (lips_aligned[i] > teeth_aligned[i] or 
                close[i] > teeth_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0