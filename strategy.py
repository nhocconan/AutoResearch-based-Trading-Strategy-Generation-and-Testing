#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams Alligator + 1d Elder Ray + volume confirmation
# Williams Alligator identifies trend direction using smoothed medians (Jaw/Teeth/Lips)
# Elder Ray measures bull/bear power via EMA(13) relative to high/low
# Long when: Alligator bullish (Lips>Teeth>Jaw) AND Bull Power > 0 AND volume > 1.5x MA(20)
# Short when: Alligator bearish (Lips<Teeth<Jaw) AND Bear Power < 0 AND volume > 1.5x MA(20)
# Uses discrete position sizing 0.25 to target ~30-50 trades/year and minimize fee drag
# Works in bull/bear markets: Alligator filters chop, Elder Ray confirms momentum, volume avoids false breakouts

name = "4h_12h_1d_alligator_elder_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Williams Alligator on 12h ===
    # Smoothed medians: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
    median_12h = (high_12h + low_12h) / 2.0
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === Elder Ray on 1d ===
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current 4h volume > 1.5x average 4h volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if Alligator turns bearish OR Bull Power <= 0
            if (lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i] or
                bull_power_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Alligator turns bullish OR Bear Power >= 0
            if (lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i] or
                bear_power_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Alligator bullish AND Bull Power > 0 AND volume confirmed
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0 and volume_confirmed[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: Alligator bearish AND Bear Power < 0 AND volume confirmed
            elif (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and
                  bear_power_aligned[i] < 0 and volume_confirmed[i]):
                position = -1
                signals[i] = -0.25
    
    return signals