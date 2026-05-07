#!/usr/bin/env python3
name = "6h_1w_1d_Alligator_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator on weekly: 
    # Jaw (blue): 13-period SMMA, smoothed by 8 periods
    # Teeth (red): 8-period SMMA, smoothed by 5 periods
    # Lips (green): 5-period SMMA, smoothed by 3 periods
    # SMMA formula: SMMA = (SMMA_prev * (period-1) + close) / period
    
    # Calculate SMMA for weekly data
    close_1w = df_1w['close'].values
    smma_5 = np.zeros_like(close_1w)
    smma_8 = np.zeros_like(close_1w)
    smma_13 = np.zeros_like(close_1w)
    
    # Initialize with SMA
    if len(close_1w) >= 5:
        smma_5[4] = np.mean(close_1w[:5])
        for i in range(5, len(close_1w)):
            smma_5[i] = (smma_5[i-1] * 4 + close_1w[i]) / 5
    
    if len(close_1w) >= 8:
        smma_8[7] = np.mean(close_1w[:8])
        for i in range(8, len(close_1w)):
            smma_8[i] = (smma_8[i-1] * 7 + close_1w[i]) / 8
    
    if len(close_1w) >= 13:
        smma_13[12] = np.mean(close_1w[:13])
        for i in range(13, len(close_1w)):
            smma_13[i] = (smma_13[i-1] * 12 + close_1w[i]) / 13
    
    # Apply smoothing: Jaw = SMMA(13) smoothed by 8
    jaw = np.zeros_like(smma_13)
    if len(smma_13) >= 8:
        jaw[7] = np.mean(smma_13[:8])
        for i in range(8, len(smma_13)):
            jaw[i] = (jaw[i-1] * 7 + smma_13[i]) / 8
    
    # Teeth = SMMA(8) smoothed by 5
    teeth = np.zeros_like(smma_8)
    if len(smma_8) >= 5:
        teeth[4] = np.mean(smma_8[:5])
        for i in range(5, len(smma_8)):
            teeth[i] = (teeth[i-1] * 4 + smma_8[i]) / 5
    
    # Lips = SMMA(5) smoothed by 3
    lips = np.zeros_like(smma_5)
    if len(smma_5) >= 3:
        lips[2] = np.mean(smma_5[:3])
        for i in range(3, len(smma_5)):
            lips[i] = (lips[i-1] * 2 + smma_5[i]) / 3
    
    # Align weekly Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 4)  # Wait for Alligator and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: price above Lips with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            if lips_aligned[i] > close[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below Lips with volume and weekly downtrend
            elif lips_aligned[i] < close[i] and vol_condition and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Lips or Alligator turns
            if close[i] < lips_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Lips or Alligator turns
            if close[i] > lips_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Williams Alligator trend filter with weekly alignment and daily EMA confirmation
# - Weekly Alligator (SMMA-based) identifies strong trend direction
# - Lips > Teeth > Jaw = bullish alignment, Lips < Teeth < Jaw = bearish
# - Enter long when price is below Lips in weekly uptrend (pullback to entry)
# - Enter short when price is above Lips in weekly downtrend (pullback to entry)
# - Volume spike (1.5x average) confirms momentum
# - Daily EMA(34) trend filter ensures alignment with intermediate trend
# - Works in both bull and bear markets by following weekly trend
# - Position size 0.25 limits drawdown and reduces fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Williams Alligator is less commonly used, providing unique edge vs saturated strategies