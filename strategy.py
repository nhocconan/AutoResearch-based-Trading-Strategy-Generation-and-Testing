#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend presence via convergence/divergence
# In bear markets, Alligator lines converge (sleeping) - wait for divergence (awakening) with volume
# 1d EMA(34) ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA) confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator calculation (using 12h data as primary timeframe)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    close_series = pd.Series(close)
    jaw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = close_series.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Alligator sleeping: Jaw, Teeth, Lips intertwined (convergence)
        # Alligator awake: Lines diverging with clear separation
        jaw_above_teeth = jaw[i] > teeth[i]
        teeth_above_lips = teeth[i] > lips[i]
        jaw_below_teeth = jaw[i] < teeth[i]
        teeth_below_lips = teeth[i] < lips[i]
        
        # Long: Alligator awake (jaw > teeth > lips) + price above 1d EMA34 + volume spike
        # Short: Alligator awake (jaw < teeth < lips) + price below 1d EMA34 + volume spike
        if position == 0:
            if jaw_above_teeth and teeth_above_lips and close[i] > ema_34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif jaw_below_teeth and teeth_below_lips and close[i] < ema_34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator starts sleeping (lines converge) OR price below 1d EMA34
            if not (jaw_above_teeth and teeth_above_lips) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator starts sleeping (lines converge) OR price above 1d EMA34
            if not (jaw_below_teeth and teeth_below_lips) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals