#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses 1d Camarilla R3/S3 levels for breakout entries, filtered by 1d trend (EMA50) and volume spikes (>1.5x 20-period average).
# Trades only when price breaks R3 (long) or S3 (short) on 12h timeframe with volume confirmation.
# Designed for low trade frequency (<150 total 12h trades) to minimize fee drag.
# Works in bull/bear markets by following 1d trend direction while using Camarilla levels for precise entries.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # We use R3 and S3 for breakouts
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous, set to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.0 * (prev_high - prev_low)
    S3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # 1d trend filter: EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_long = close_1d > ema_50   # Uptrend filter
    trend_short = close_1d < ema_50  # Downtrend filter
    
    # Align 1d indicators to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    trend_long_aligned = align_htf_to_ltf(prices, df_1d, trend_long)
    trend_short_aligned = align_htf_to_ltf(prices, df_1d, trend_short)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(trend_long_aligned[i]) or np.isnan(trend_short_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + uptrend + volume spike
            if (close[i] > R3_aligned[i] and 
                trend_long_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + downtrend + volume spike
            elif (close[i] < S3_aligned[i] and 
                  trend_short_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR trend turns down
            if (close[i] < S3_aligned[i]) or \
               not trend_long_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR trend turns up
            if (close[i] > R3_aligned[i]) or \
               not trend_short_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals