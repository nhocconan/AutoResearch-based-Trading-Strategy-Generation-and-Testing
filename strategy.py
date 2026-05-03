#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h for signal direction and 1h for precise entry timing. Session filter (08-20 UTC) reduces noise.
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours once (avoid datetime ops in loop)
    hours = prices.index.hour
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar
    typical_price = (high + low + close) / 3.0
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan
    
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev)
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Session filter: 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 4h trend filter
        if position == 0:
            if close[i] > camarilla_r3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            elif close[i] < camarilla_s3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 OR below 4h EMA50
            if close[i] < camarilla_s3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R3 OR above 4h EMA50
            if close[i] > camarilla_r3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals