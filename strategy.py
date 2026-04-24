#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
- Williams Alligator: Jaw (EMA13, 8-period smoothed), Teeth (EMA8, 5-period smoothed), Lips (EMA5, 3-period smoothed)
- Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and price above 1d EMA34
- Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and price below 1d EMA34
- 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws
- Volume spike (>2.0x 30-period average) confirms conviction
- Designed for low trade frequency (target: 12-37/year) to minimize fee drag on 12h timeframe
- Works in bull/bear via trend filter and clear Alligator alignment signals
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (using 12h data directly)
    # Jaw: EMA13 of median price, smoothed by 8 periods
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA8 of median price, smoothed by 5 periods
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA5 of median price, smoothed by 3 periods
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips_raw).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: > 2.0x 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 13, 8, 5, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment (Lips > Teeth > Jaw) with volume spike and price above daily EMA34
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume_spike[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment (Lips < Teeth < Jaw) with volume spike and price below daily EMA34
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  volume_spike[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth OR Teeth <= Jaw)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth OR Teeth >= Jaw)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0