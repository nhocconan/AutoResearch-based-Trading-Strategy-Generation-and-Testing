#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels from 1d provide key support/resistance, 1d EMA34 filters for higher timeframe trend alignment,
# volume spike confirms institutional participation. Designed for 12-30 trades/year to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
# Uses 12h primary timeframe as specified in experiment #124316.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Camarilla levels from prior completed 1d bar
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    rng = high_1d - low_1d
    camarilla_R3 = close_1d + 1.1 * rng / 2
    camarilla_S3 = close_1d - 1.1 * rng / 2
    
    # Shift to use only prior completed 1d bar
    camarilla_R3_shifted = np.roll(camarilla_R3, 1)
    camarilla_S3_shifted = np.roll(camarilla_S3, 1)
    camarilla_R3_shifted[0] = np.nan
    camarilla_S3_shifted[0] = np.nan
    
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_shifted)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Camarilla R3 AND 1d EMA34 uptrend AND volume spike
            if close[i] > camarilla_R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Camarilla S3 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < camarilla_S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR below 1d EMA34
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR above 1d EMA34
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals