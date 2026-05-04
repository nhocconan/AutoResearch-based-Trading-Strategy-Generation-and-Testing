#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation
# Alligator identifies trendless markets (sleeping) vs trending (awake). Enter when Lips cross Teeth/Jaw in trend direction.
# 1d EMA50 ensures alignment with higher timeframe trend. Volume spike (>1.5x 20 EMA) filters weak breakouts.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 50-150 trades over 4 years.

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h timeframe: SMA of median price
    # Jaw: 13-period SMA, 8 bars ahead
    # Teeth: 8-period SMA, 5 bars ahead  
    # Lips: 5-period SMA, 3 bars ahead
    median_price = (high + low) / 2.0
    median_series = pd.Series(median_price)
    
    jaw = median_series.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = median_series.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = median_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: Lips cross above Teeth AND Teeth above Jaw (bullish alignment) + uptrend + volume
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema50_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips cross below Teeth AND Teeth below Jaw (bearish alignment) + downtrend + volume
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema50_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator sleeping (Lips < Teeth < Jaw) OR trend changes OR volume drops
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or \
               close[i] < ema50_aligned[i] or \
               volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator sleeping (Lips > Teeth > Jaw) OR trend changes OR volume drops
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or \
               close[i] > ema50_aligned[i] or \
               volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals