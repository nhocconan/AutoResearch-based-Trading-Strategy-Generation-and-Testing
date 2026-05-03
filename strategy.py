#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(34) trend filter and volume confirmation
# Designed to capture breakouts aligned with intermediate trend while filtering low-momentum noise.
# Uses 4h/1d for signal direction (trend/volume), 1h only for entry timing precision.
# Session filter (08-20 UTC) reduces off-hours noise. Discrete position sizing (0.20) minimizes fee drag.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h. Works in bull/bear by following 4h EMA direction.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute session hours (08-20 UTC) once
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 4h bar (HLC of completed 4h bar)
    # Camarilla R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Use prior completed 4h bar (shift 1) to avoid look-ahead
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan  # First value undefined
    camarilla_s3_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_shifted)
    
    # Calculate 4h EMA(34) for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: 20-period EMA on 1h volume (same timeframe)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid EMA and volume
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tighten to reduce trades)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = close[i] > ema_34_4h_aligned[i]
        price_below_ema = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 4h EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and price_above_ema and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + below 4h EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and price_below_ema and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses 4h trend alignment
            if close[i] < camarilla_s3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses 4h trend alignment
            if close[i] > camarilla_r3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals