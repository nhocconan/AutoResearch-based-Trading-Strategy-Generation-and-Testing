#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla R3/S3 levels provide high-probability breakout zones. 1d EMA34 ensures alignment with higher timeframe trend.
# Volume spike (>2x 20 EMA) confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Discrete sizing 0.20 limits risk. Works in bull/bear: trend filter prevents counter-trend entries.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_R3S3_1dEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC) - prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h timeframe (using previous completed 4h bar)
    # Camarilla: based on previous day's (here: previous 4h bar) range
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Typical price for pivot calculation
    typical_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla levels: R3, S3 (most significant breakout levels)
    # R3 = close + 1.1 * range * 1.1/4 = close + 1.1 * range * 0.275
    # S3 = close - 1.1 * range * 1.1/4 = close - 1.1 * range * 0.275
    camarilla_r3 = typical_4h + (1.1 * range_4h * 0.275)
    camarilla_s3 = typical_4h - (1.1 * range_4h * 0.275)
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 1h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot level OR trend changes OR volume drops
            camarilla_pivot = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2.0
            if (close[i] < camarilla_pivot or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla pivot level OR trend changes OR volume drops
            camarilla_pivot = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2.0
            if (close[i] > camarilla_pivot or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals