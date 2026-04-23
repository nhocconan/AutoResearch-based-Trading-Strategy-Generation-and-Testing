#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Williams Alligator combination with volume confirmation.
- Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for momentum
- 1d Williams Alligator (Jaw=TEETH=13, Teeth=TEETH=8, Lips=TEETH=5 SMAs) for trend direction
- Long when Bull Power > 0 and price above Alligator Teeth (uptrend)
- Short when Bear Power > 0 and price below Alligator Teeth (downtrend)
- Volume > 1.5x 20-period average for confirmation
- Position size: 0.25 discrete level
- Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
- Works in bull/bear via Alligator trend filter + Elder Ray momentum
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_6h = high_6h - ema_13_6h  # Bull Power = High - EMA13
    bear_power_6h = ema_13_6h - low_6h   # Bear Power = EMA13 - Low
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator SMAs (using close prices)
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Alligator trend: price above teeth = uptrend, below teeth = downtrend
    # Alligator sleeping: jaws, teeth, lips intertwined (no clear trend)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5)  # Volume MA, Alligator periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator trend conditions
        # Uptrend: price above teeth and teeth above lips
        uptrend = close[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        # Downtrend: price below teeth and teeth below lips
        downtrend = close[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND uptrend AND volume confirmation
            if bull_power_aligned[i] > 0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND downtrend AND volume confirmation
            elif bear_power_aligned[i] > 0 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price crosses below teeth (trend change)
            if bull_power_aligned[i] <= 0 or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 OR price crosses above teeth (trend change)
            if bear_power_aligned[i] <= 0 or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0