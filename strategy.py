#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ATR Regime Filter and Volume Spike.
- Primary timeframe: 12h for execution, HTF: 1d for ATR regime and Williams Alligator.
- Entry: Williams Alligator signals bullish/bearish alignment (jaw/teeth/lips) with volume > 2.0x 20-period volume MA.
- Regime filter: Only trade when 1d ATR(14) > 1.5 * ATR(50) (high volatility regime) to avoid chop.
- Williams Alligator: Jaw (SMA13 of median price, shifted 8), Teeth (SMA8 of median price, shifted 5), Lips (SMA5 of median price, shifted 3).
- Bullish: Lips > Teeth > Jaw. Bearish: Lips < Teeth < Jaw.
- Exit: Reverse Alligator signal or volatility contraction (ATR ratio < 1.0).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying Alligator alignment in high vol, in bear via selling alignment in high vol.
- Avoids overtrading by requiring strong volatility regime and clear Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # > 1.5 indicates high volatility regime
    
    # Align ATR ratio to 12h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Williams Alligator on 1d: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Median price = (high + low) / 2
    median_price_1d = (high_1d + low_1d) / 2
    
    # Jaw: SMA13 of median price, shifted 8 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw_1d = np.concatenate([np.full(8, np.nan), jaw_1d[:-8]])  # shift right by 8
    
    # Teeth: SMA8 of median price, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = np.concatenate([np.full(5, np.nan), teeth_1d[:-5]])  # shift right by 5
    
    # Lips: SMA5 of median price, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = np.concatenate([np.full(3, np.nan), lips_1d[:-3]])  # shift right by 3
    
    # Align Alligator lines to 12h timeframe (completed 1d bar only)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3) + 1  # Need ATR50, volume MA, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in high volatility (ATR ratio > 1.5)
        in_high_vol = atr_ratio_aligned[i] > 1.5
        
        if position == 0:
            # Long: Bullish Alligator alignment (Lips > Teeth > Jaw) + volume spike + high vol
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                volume_spike[i] and in_high_vol):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment (Lips < Teeth < Jaw) + volume spike + high vol
            elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                  volume_spike[i] and in_high_vol):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR volatility contraction (ATR ratio < 1.0)
            if (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] or 
                atr_ratio_aligned[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR volatility contraction (ATR ratio < 1.0)
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] or 
                atr_ratio_aligned[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0