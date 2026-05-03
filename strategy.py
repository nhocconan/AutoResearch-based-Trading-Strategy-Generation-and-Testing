#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Camarilla levels provide precise intraday support/resistance based on prior day's range.
# R3/S3 are strong reversal levels; breaks above R3/below S3 with volume indicate institutional participation.
# 12h EMA34 filters for medium-term trend alignment to avoid counter-trend whipsaws.
# Volume confirmation (2.0x 20-period EMA) ensures breakouts have conviction.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in direction of 12h trend.

name = "6h_Camarilla_R3S3_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA34 trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from prior 12h bar
    # Camarilla: based on previous period's high, low, close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r3 = np.full(len(close_12h), np.nan)
    camarilla_s3 = np.full(len(close_12h), np.nan)
    camarilla_r4 = np.full(len(close_12h), np.nan)
    camarilla_s4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):  # Start from 1 to have prior bar
        # Prior 12h bar values
        phigh = high_12h[i-1]
        plow = low_12h[i-1]
        pclose = close_12h[i-1]
        range_val = phigh - plow
        
        if range_val <= 0:
            camarilla_r3[i] = camarilla_s3[i] = camarilla_r4[i] = camarilla_s4[i] = np.nan
        else:
            camarilla_r3[i] = pclose + range_val * 1.1 / 4
            camarilla_s3[i] = pclose - range_val * 1.1 / 4
            camarilla_r4[i] = pclose + range_val * 1.1 / 2
            camarilla_s4[i] = pclose - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Medium-term trend: price above/below 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in uptrend
            if close[i] > camarilla_r3_aligned[i] and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend
            elif close[i] < camarilla_s3_aligned[i] and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend
            if close[i] < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend
            if close[i] > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals