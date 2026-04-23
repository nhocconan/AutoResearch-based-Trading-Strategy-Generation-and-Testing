#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray + Volume Confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) defines trend: 
  Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Enter long when: Alligator bullish AND Bull Power > 0 AND volume > 1.5x average
- Enter short when: Alligator bearish AND Bear Power > 0 AND volume > 1.5x average
- Exit when Alligator trend reverses OR power fails
- Uses 1d EMA34 as higher timeframe trend filter to avoid counter-trend trades
- Discrete position size 0.25 to manage drawdown in volatile 6h markets
- Target: 12-25 trades/year on 6h (50-100 total over 4 years)
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
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Jaw: 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Teeth: 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Lips: 5-period
    
    # Elder Ray Power: using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA34 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 34)  # volume MA, Alligator jaws, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator trend
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power > 0 AND price above 1d EMA34 AND volume
            if alligator_bullish and bull_power[i] > 0 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power > 0 AND price below 1d EMA34 AND volume
            elif alligator_bearish and bear_power[i] > 0 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power <= 0 OR price below 1d EMA34
            if not alligator_bullish or bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power <= 0 OR price above 1d EMA34
            if not alligator_bearish or bear_power[i] <= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0