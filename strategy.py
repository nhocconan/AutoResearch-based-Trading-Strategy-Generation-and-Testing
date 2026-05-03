#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h for signal direction (trend filter and Camarilla levels) and 1h only for entry timing precision
# Volume spike (>2.0x 20-period EMA on 1h) filters low-probability breakouts
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag
# Discrete position sizing (0.20) minimizes fee churn while maintaining exposure

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
    
    # Get 4h data for HTF indicators (trend filter and Camarilla levels)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    # Shift to use previous bar's values (no look-ahead)
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    close_4h_prev = np.roll(close_4h_arr, 1)
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    close_4h_prev[0] = np.nan
    
    camarilla_r3_4h = close_4h_prev + 1.1 * (high_4h_prev - low_4h_prev)
    camarilla_s3_4h = close_4h_prev - 1.1 * (high_4h_prev - low_4h_prev)
    
    # Align HTF indicators to 1h timeframe
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime operations in loop)
    # open_time is already datetime64[ms], so we can use .hour directly on the index
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 4h trend filter
        # Long: Break above R3 + price above 4h EMA50 + volume spike
        # Short: Break below S3 + price below 4h EMA50 + volume spike
        if position == 0:
            if close[i] > camarilla_r3_4h_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            elif close[i] < camarilla_s3_4h_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversion to mean) OR below 4h EMA50
            if close[i] < camarilla_s3_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R3 (reversion to mean) OR above 4h EMA50
            if close[i] > camarilla_r3_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals