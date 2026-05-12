#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
Works in bull/bear markets because: 1) Alligator identifies trend via Jaw/Teeth/Lips alignment (trending when Lips > Teeth > Jaw for long, reverse for short), 2) 1d EMA50 ensures alignment with higher timeframe trend, 3) Volume spike confirms momentum, reducing false signals. Target 15-30 trades/year.
"""
name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1d DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === WILLIAMS ALLIGATOR (13,8,5 SMoothed) ===
    # Jaw (13-period SMMA, 8-bar shift)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, 5-bar shift)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, 3-bar shift)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align to 12h timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 8, 5, 20)  # 50 for daily EMA, 13/8/5 for Alligator, 20 for volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment), price above daily EMA50, volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment), price below daily EMA50, volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Lips cross below Teeth or price below daily EMA50
            if (lips_aligned[i] < teeth_aligned[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips cross above Teeth or price above daily EMA50
            if (lips_aligned[i] > teeth_aligned[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals