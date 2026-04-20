#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams Alligator components (Bill Williams)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw (13-period SMMA of median price, shifted 8 bars)
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # Shift forward by 8 bars
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth (8-period SMMA of median price, shifted 5 bars)
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # Shift forward by 5 bars
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips (5-period SMMA of median price, shifted 3 bars)
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # Shift forward by 3 bars
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_12h = vol_ratio[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Volatility filter: avoid extreme volatility
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_12h > 1.5)
        
        if position == 0:
            # Enter long when Alligator is bullish aligned with volume and volatility filter
            if bullish_alignment and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when Alligator is bearish aligned with volume and volatility filter
            elif bearish_alignment and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks or volatility spike
            if not bullish_alignment or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks or volatility spike
            if not bearish_alignment or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0