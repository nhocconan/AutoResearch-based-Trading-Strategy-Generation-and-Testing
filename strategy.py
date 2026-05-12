#!/usr/bin/env python3
"""
6h_Volume_Regime_Breakout_12hTrend
Hypothesis: Uses 6h timeframe with breakout from 20-period high/low, filtered by 12h trend (via EMA50) and volume regime (low volatility followed by spike). 
This strategy aims to capture momentum bursts after periods of consolidation, working in both bull and bear markets by aligning with the 12h trend.
Only takes long when price breaks above 20-period high in 12h uptrend with volume spike after low volatility.
Only takes short when price breaks below 20-period low in 12h downtrend with volume spike after low volatility.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "6h_Volume_Regime_Breakout_12hTrend"
timeframe = "6h"
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

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate 20-period high/low for breakout levels (using close-based donchian for simplicity)
    period20_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume indicators: 20-period average and volatility regime
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std_20 = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    # Low volatility regime: when volatility is below 50-period average
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = vol_std_20 < (vol_avg_50 * 0.5)  # volatility less than half of 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start from 50 to have enough data for all indicators
        # Get aligned values for current 6h bar
        ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)[i]
        vol_avg_val = vol_avg_20[i]
        low_vol = low_vol_regime[i]
        
        # Skip if any required data is NaN
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(ema50_aligned) or np.isnan(vol_avg_val) or np.isnan(low_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 20-period high + 12h uptrend + low vol regime + volume spike
            if (close[i] > period20_high[i] and 
                close[i] > ema50_aligned and 
                low_vol and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + 12h downtrend + low vol regime + volume spike
            elif (close[i] < period20_low[i] and 
                  close[i] < ema50_aligned and 
                  low_vol and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low or trend turns down
            if (close[i] < period20_low[i] or close[i] < ema50_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high or trend turns up
            if (close[i] > period20_high[i] or close[i] > ema50_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals