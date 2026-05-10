#!/usr/bin/env python3
# 1D_HMA_Trend_Filter_With_Williams_Alligator_Signal
# Hypothesis: Uses HMA on 1d for trend direction and Williams Alligator on 1d for entry timing, with volume confirmation.
# Designed for low trade frequency (10-20/year) to avoid fee drag. Works in bull/bear markets by combining trend and momentum.
# HMA(21) determines trend; Alligator (Jaw/Teeth/Lips) gives entry when aligned with trend. Volume > 1.5x average confirms.
# Position size 0.25 for balanced risk. Target: 15-25 trades/year.

name = "1D_HMA_Trend_Filter_With_Williams_Alligator_Signal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hull_moving_average(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma1 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma2 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw = 2 * wma1 - wma2
    hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def williams_alligator(high, low, close):
    """Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs shifted"""
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    # Shift as per Williams: Jaw by 8, Teeth by 5, Lips by 3
    jaw = jaw.shift(8)
    teeth = teeth.shift(5)
    lips = lips.shift(3)
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate HMA for trend (21 period)
    hma_21 = hull_moving_average(df_1d['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate Williams Alligator
    jaw, teeth, lips = williams_alligator(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 13, 20)  # Warmup for HMA, Alligator, volume
    
    for i in range(start_idx, n):
        if np.isnan(hma_21_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from HMA
        uptrend = close[i] > hma_21_aligned[i]
        downtrend = close[i] < hma_21_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw for uptrend, reverse for downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: HMA uptrend + Alligator aligned long + volume
            if uptrend and alligator_long and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: HMA downtrend + Alligator aligned short + volume
            elif downtrend and alligator_short and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend change or Alligator misalignment
            if not uptrend or not alligator_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend change or Alligator misalignment
            if not downtrend or not alligator_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals