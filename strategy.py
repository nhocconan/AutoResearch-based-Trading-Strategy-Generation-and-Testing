#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Combines Camarilla pivot levels (R1/S1) from daily timeframe with 4h price action, daily EMA trend filter, and volume confirmation. 
Buys when price breaks above R1 with daily uptrend and volume spike; sells when price breaks below S1 with daily downtrend and volume spike.
Uses tight entry conditions to limit trades (target: 20-40/year) and avoid fee drag. Works in bull/bear markets by requiring daily trend alignment.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla levels and trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align to 4h timeframe (available after daily bar closes)
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1) or np.isnan(s1) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + daily uptrend + volume confirmation
            if close[i] > r1 and close[i-1] <= r1 and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + daily downtrend + volume confirmation
            elif close[i] < s1 and close[i-1] >= s1 and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or closes below daily EMA34
            if close[i] < s1 or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or closes above daily EMA34
            if close[i] > r1 or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals