#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator trend following with 1w ADX regime filter and volume confirmation.
- Uses Alligator (Jaw/Teeth/Lips) from 6h timeframe to identify trend direction
- 1w ADX > 25 filters for strong trending regimes (avoids choppy markets)
- Volume > 1.5x 20-period average confirms breakout strength
- Long: Lips > Teeth > Jaw (bullish alignment) AND ADX > 25 AND volume confirmation
- Short: Lips < Teeth < Jaw (bearish alignment) AND ADX > 25 AND volume confirmation
- Exit: Opposite Alligator alignment OR ADX < 20 (regime change to ranging)
- Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
- Works in bull (catch trends) and bear (avoid whipsaws via ADX filter)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h Alligator components (Jaw=13, Teeth=8, Lips=5)
    # Jaw: Blue line (13-period SMMA shifted 8 bars)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    # Teeth: Red line (8-period SMMA shifted 5 bars)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    # Lips: Green line (5-period SMMA shifted 3 bars)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Calculate 1w ADX for regime filtering (trend strength)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1w).diff()
    down_move = pd.Series(low_1w).diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    
    # Smoothed DM and TR
    atr_1w_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w_smooth)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, prices, lips.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for Alligator shifts, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # ADX regime filter
        strong_trend = adx_1w_aligned[i] > 25
        ranging_market = adx_1w_aligned[i] < 20
        
        if position == 0:
            # Enter long: bullish Alligator alignment AND strong trend AND volume confirmation
            if bullish_alignment and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator alignment AND strong trend AND volume confirmation
            elif bearish_alignment and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR ranging market (regime change)
            if bearish_alignment or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR ranging market (regime change)
            if bullish_alignment or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_Trend_1wADX_VolumeConfirm"
timeframe = "6h"
leverage = 1.0