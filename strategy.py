#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume spike confirmation.
# Long when Alligator jaws-teeth-lips are aligned bullish (jaws < teeth < lips) AND Elder Bull Power > 0 AND volume > 1.5x 24-bar average.
# Short when Alligator aligned bearish (jaws > teeth > lips) AND Elder Bear Power < 0 AND volume > 1.5x 24-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture medium-term trends with low trade frequency.
# Williams Alligator identifies trend alignment via three smoothed SMAs (13,8,5). Elder Ray measures bull/bear power relative to EMA13.
# Volume spike requirement reduces false breakouts and improves signal quality.
# Works in both bull and bear markets by following the trend defined by Alligator alignment.

name = "12h_WilliamsAlligator_1dElderRay_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    close_12h = df_12h['close'].values
    # Jaw: 13-period SMA
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMA
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMA
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator alignment: bullish when jaw < teeth < lips, bearish when jaw > teeth > lips
    alligator_bullish = (jaw_aligned < teeth_aligned) & (teeth_aligned < lips_aligned)
    alligator_bearish = (jaw_aligned > teeth_aligned) & (teeth_aligned > lips_aligned)
    
    # Load 1d data ONCE before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray components to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current 12h volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Bull Power > 0 AND volume confirmation
            if (alligator_bullish[i] and 
                bull_power_aligned[i] > 0 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND volume confirmation
            elif (alligator_bearish[i] and 
                  bear_power_aligned[i] < 0 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Bull Power <= 0
            if (not alligator_bullish[i] or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Bear Power >= 0
            if (not alligator_bearish[i] or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals