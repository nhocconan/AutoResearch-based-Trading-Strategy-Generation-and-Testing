#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume confirmation
# Long when Alligator is bullish (jaw < teeth < lips) + Elder Ray bull power > 0 + volume > 1.5x avg
# Short when Alligator is bearish (jaw > teeth > lips) + Elder Ray bear power < 0 + volume > 1.5x avg
# Uses 12h timeframe for lower trade frequency (target: 12-37 trades/year) to minimize fee drag
# Alligator identifies trend, Elder Ray measures bull/bear power, volume confirms strength
# Designed to work in both bull and bear markets by capturing strong directional moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Williams Alligator (13,8,5) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Jaw (13-period SMMA)
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMMA)
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMMA)
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # === 1d Indicators: Elder Ray (13-period EMA) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA of close
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema_13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or
            np.isnan(lips_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish: jaw < teeth < lips
        # 2. Elder Ray bull power > 0 (buying pressure)
        # 3. Volume confirmation
        if (jaw_12h_aligned[i] < teeth_12h_aligned[i] < lips_12h_aligned[i]) and \
           (bull_power_aligned[i] > 0) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish: jaw > teeth > lips
        # 2. Elder Ray bear power < 0 (selling pressure)
        # 3. Volume confirmation
        elif (jaw_12h_aligned[i] > teeth_12h_aligned[i] > lips_12h_aligned[i]) and \
             (bear_power_aligned[i] < 0) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Alligator_ElderRay_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0