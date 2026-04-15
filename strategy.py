#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator Jaw (teeth > lips) + 1w close > 1w EMA34 + volume > 1.5x 20-period avg
# Short when price < Alligator Jaw (teeth < lips) + 1w close < 1w EMA34 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# Williams Alligator (SMMA13,8,5) identifies trend initiation and continuation.
# 1w EMA34 provides strong higher-timeframe trend filter reducing whipsaws.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 1d timeframe.
# Works in both bull and bear markets by following the higher-timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Williams Alligator (SMMA13,8,5) ===
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value: SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)   # Blue line
    teeth = smma(close_1d, 8)   # Red line
    lips = smma(close_1d, 5)    # Green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1w HTF: EMA34 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20) + 5  # EMA34 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Alligator condition: Jaw position indicates trend
        # Jaw > Teeth > Lips = uptrend (green/red/blue from bottom to top)
        # Jaw < Teeth < Lips = downtrend (blue/red/green from bottom to top)
        jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
        teeth_above_lips = teeth_aligned[i] > lips_aligned[i]
        jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
        teeth_below_lips = teeth_aligned[i] < lips_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price > Jaw (trend confirmation)
        # 2. Jaw > Teeth > Lips (uptrend alignment)
        # 3. 1w EMA34 uptrend (close > EMA34)
        # 4. Volume confirmation
        if (close[i] > jaw_aligned[i]) and \
           jaw_above_teeth and teeth_above_lips and \
           (close_1w[-1] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price < Jaw (trend confirmation)
        # 2. Jaw < Teeth < Lips (downtrend alignment)
        # 3. 1w EMA34 downtrend (close < EMA34)
        # 4. Volume confirmation
        elif (close[i] < jaw_aligned[i]) and \
             jaw_below_teeth and teeth_below_lips and \
             (close_1w[-1] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Williams_Alligator_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0