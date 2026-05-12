#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 4h timeframe with breakouts from Camarilla R3/S3 levels (derived from 1d high/low/close),
filtered by 1d EMA34 trend and volume spike after low volatility.
Captures momentum bursts after consolidation, aligned with daily trend.
Works in both bull and bear markets by following the 1d trend direction.
Uses discrete position sizing (0.25) to minimize fee churn.
Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and Camarilla levels (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from 1d OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    daily_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + daily_range * 1.1 / 2
    camarilla_s3 = df_1d['close'] - daily_range * 1.1 / 2
    
    # Volume indicators: 20-period average and volatility regime
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std_20 = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = vol_std_20 < (vol_avg_50 * 0.5)  # volatility less than half of 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start from 50 to have enough data for all indicators
        # Get aligned values for current 4h bar
        ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)[i]
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)[i]
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)[i]
        vol_avg_val = vol_avg_20[i]
        low_vol = low_vol_regime[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned) or np.isnan(camarilla_r3_aligned) or 
            np.isnan(camarilla_s3_aligned) or np.isnan(vol_avg_val) or np.isnan(low_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 + 1d uptrend + low vol regime + volume spike
            if (close[i] > camarilla_r3_aligned and 
                close[i] > ema34_aligned and 
                low_vol and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + 1d downtrend + low vol regime + volume spike
            elif (close[i] < camarilla_s3_aligned and 
                  close[i] < ema34_aligned and 
                  low_vol and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 or trend turns down
            if (close[i] < camarilla_s3_aligned or close[i] < ema34_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 or trend turns up
            if (close[i] > camarilla_r3_aligned or close[i] > ema34_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals