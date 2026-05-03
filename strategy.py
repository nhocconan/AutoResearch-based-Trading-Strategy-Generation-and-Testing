#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(34) trend filter and volume confirmation
# R3/S3 are stronger Camarilla levels than R1/S1, providing higher probability breakouts
# 12h EMA(34) is more responsive than EMA(50) while still filtering counter-trend moves
# Volume confirmation (>1.8x 20-period EMA) reduces false breakouts
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fee drag

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = Close + 1.1*(High-Low)/4, S3 = Close - 1.1*(High-Low)/4
    camarilla_range = (high_1d - low_1d)
    r3 = close_1d + (1.1 * camarilla_range / 4)
    s3 = close_1d - (1.1 * camarilla_range / 4)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Camarilla breakout signals with 12h trend filter
        # Long: Break above R3 + price above 12h EMA34 + volume spike
        # Short: Break below S3 + price below 12h EMA34 + volume spike
        if position == 0:
            if close[i] > r3_aligned[i] and close[i] > ema_34_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < s3_aligned[i] and close[i] < ema_34_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversion to mean) OR below 12h EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 (reversion to mean) OR above 12h EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals