#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot filter and volume confirmation
# Uses weekly Camarilla pivot levels (R3/S3) to align with major weekly structure
# Donchian breakout provides trend entry, weekly pivot acts as regime filter
# Volume confirmation reduces false breakouts. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via upward breaks above weekly R3 and in bear markets via downward breaks below weekly S3.

name = "6h_Donchian20_WeeklyCamarillaR3S3_VolumeSpike"
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) for trend context
    high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20 = high_20
    donchian_low_20 = low_20
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate weekly Camarilla levels from previous weekly bar
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    weekly_R3 = prev_weekly_close + 1.1 * (prev_weekly_high - prev_weekly_low)
    weekly_S3 = prev_weekly_close - 1.1 * (prev_weekly_high - prev_weekly_low)
    
    # Align weekly Camarilla levels to 6h timeframe (use previous week's levels)
    weekly_R3_aligned = align_htf_to_ltf(prices, df_1w, weekly_R3)
    weekly_S3_aligned = align_htf_to_ltf(prices, df_1w, weekly_S3)
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(weekly_R3_aligned[i]) or np.isnan(weekly_S3_aligned[i]) or
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above 6h Donchian high AND above weekly R3 with volume spike
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > weekly_R3_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian low AND below weekly S3 with volume spike
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < weekly_S3_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 6h Donchian low or below weekly S3
            if close[i] < donchian_low_20_aligned[i] or close[i] < weekly_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 6h Donchian high or above weekly R3
            if close[i] > donchian_high_20_aligned[i] or close[i] > weekly_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals