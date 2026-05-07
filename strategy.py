#!/usr/bin/env python3
name = "12h_WilliamsAlligator_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Williams Alligator: SMA(13,8) Jaw, SMA(8,5) Teeth, SMA(5,3) Lips
    # Jaw: 13-period SMA, smoothed 8 bars ahead
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    # Teeth: 8-period SMA, smoothed 5 bars ahead
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    # Lips: 5-period SMA, smoothed 3 bars ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # 12h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and Alligator components
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator alignment: Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend
            uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
            downtrend = lips[i] < teeth[i] and teeth[i] < jaw[i]
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            
            # Long: Alligator aligned up + price above Lips + volume + 1d uptrend
            if uptrend and close[i] > lips[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + price below Lips + volume + 1d downtrend
            elif downtrend and close[i] < lips[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Teeth or Alligator loses alignment
            if close[i] < teeth[i] or not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Teeth or Alligator loses alignment
            if close[i] > teeth[i] or not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation
# - Williams Alligator identifies trending vs ranging markets via jaw/teeth/lips alignment
# - Only trade in direction of Alligator alignment (avoid chop)
# - 1d EMA(34) ensures alignment with higher timeframe trend
# - Volume spike (1.5x average) confirms institutional participation
# - Works in bull (buy when aligned up) and bear (sell when aligned down)
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Exit when price crosses Teeth or Alligator loses alignment provides clear signal