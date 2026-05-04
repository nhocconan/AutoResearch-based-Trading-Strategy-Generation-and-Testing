#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 12h for structure, 1w EMA34 for trend filter (proven BTC/ETH edge),
# and volume spike for confirmation. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts at R3 and in bear markets via downside breakdowns at S3.
# The 1w EMA34 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike_TrendFilter"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter from prior completed 1w bar
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_shifted = np.roll(ema34_1w, 1)
    ema34_1w_shifted[0] = np.nan
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_shifted)
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (R3, S3) from prior completed 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_h = close_12h + (1.1 * (high_12h - low_12h) / 2)
    camarilla_l = close_12h - (1.1 * (high_12h - low_12h) / 2)
    camarilla_h_shifted = np.roll(camarilla_h, 1)
    camarilla_l_shifted = np.roll(camarilla_l, 1)
    camarilla_h_shifted[0] = np.nan
    camarilla_l_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(camarilla_h_shifted[i]) or np.isnan(camarilla_l_shifted[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Camarilla R3 AND 1w EMA34 uptrend AND volume spike
            if close[i] > camarilla_h_shifted[i] and close[i] > ema34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Camarilla S3 AND 1w EMA34 downtrend AND volume spike
            elif close[i] < camarilla_l_shifted[i] and close[i] < ema34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR below 1w EMA34
            if close[i] < camarilla_l_shifted[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR above 1w EMA34
            if close[i] > camarilla_h_shifted[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals