#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivots identify key intraday support/resistance levels; breakout at R3/S3 with volume
# confirmation captures sustained moves. 1d EMA34 ensures alignment with daily trend to avoid
# counter-trend whipsaws. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by following the 1d trend direction.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from completed 1d bars
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using shift(1) to ensure we only use completed 1d bars
    prev_high = pd.Series(high_1d).shift(1).values
    prev_low = pd.Series(low_1d).shift(1).values
    prev_close = pd.Series(close_1d).shift(1).values
    
    # Calculate pivot levels
    daily_range = prev_high - prev_low
    camarilla_r3 = prev_close + (1.1 * daily_range * 1.1 / 4)
    camarilla_s3 = prev_close - (1.1 * daily_range * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid EMA and volume EMA
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 1d EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + below 1d EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses 1d trend alignment
            if close[i] < camarilla_s3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses 1d trend alignment
            if close[i] > camarilla_r3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals