#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8 periods smoothed), Teeth (EMA8, 5 periods smoothed), Lips (EMA5, 3 periods smoothed)
# Long when: Lips > Teeth > Jaw AND close > 1d EMA50 AND volume > 1.5x 20-period MA
# Short when: Jaw > Teeth > Lips AND close < 1d EMA50 AND volume > 1.5x 20-period MA
# Exit when: Alligator alignment breaks (Lips <= Teeth for long exit, Teeth <= Lips for short exit)
# Uses Alligator for trend definition, 1d EMA for higher timeframe alignment, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 12h data
    if len(close) >= 13:
        # Jaw: EMA13, smoothed by 8 periods
        jaw_raw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        jaw = pd.Series(jaw_raw).ewm(span=8, adjust=False, min_periods=8).mean().values
        
        # Teeth: EMA8, smoothed by 5 periods
        teeth_raw = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
        teeth = pd.Series(teeth_raw).ewm(span=5, adjust=False, min_periods=5).mean().values
        
        # Lips: EMA5, smoothed by 3 periods
        lips_raw = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
        lips = pd.Series(lips_raw).ewm(span=3, adjust=False, min_periods=3).mean().values
    else:
        jaw = np.full(n, np.nan)
        teeth = np.full(n, np.nan)
        lips = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND above 1d EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Jaw > Teeth > Lips (bearish alignment) AND below 1d EMA50 AND volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator bullish alignment breaks (Lips <= Teeth)
            if lips[i] <= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator bearish alignment breaks (Teeth <= Lips)
            if teeth[i] <= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals