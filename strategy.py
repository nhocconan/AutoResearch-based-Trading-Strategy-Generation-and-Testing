#!/usr/bin/env python3
# 1D_WV_R1_S1_Breakout_1W_EMA34_Trend_Volume
# Hypothesis: 1-day Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume confirmation.
# Uses weekly EMA34 to filter trend direction, reducing false breakouts in counter-trend moves.
# Volume filter ensures breakouts have participation. Designed for low trade frequency (15-25/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear markets via counter-trend bounces from S1/R1 levels.

name = "1D_WV_R1_S1_Breakout_1W_EMA34_Trend_Volume"
timeframe = "1d"
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
    
    # Calculate weekly EMA34 for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Range = high - low
    daily_range = high - low
    
    # Camarilla levels (based on previous day's action)
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # R2 = close + range * 1.1/6
    # R1 = close + range * 1.1/12
    # S1 = close - range * 1.1/12
    # S2 = close - range * 1.1/6
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    camarilla_r1 = close + daily_range * 1.1 / 12
    camarilla_s1 = close - daily_range * 1.1 / 12
    
    # Shift to get previous day's levels (to avoid look-ahead)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan
    
    # Volume confirmation (20-day average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or \
           np.isnan(camarilla_r1_prev[i]) or np.isnan(camarilla_s1_prev[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R1 with volume and weekly uptrend
            if close[i] > camarilla_r1_prev[i] and vol_confirm and ema34_1w_aligned[i] > ema34_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and weekly downtrend
            elif close[i] < camarilla_s1_prev[i] and vol_confirm and ema34_1w_aligned[i] < ema34_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 or weekly trend turns down
            if close[i] < camarilla_s1_prev[i] or ema34_1w_aligned[i] < ema34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 or weekly trend turns up
            if close[i] > camarilla_r1_prev[i] or ema34_1w_aligned[i] > ema34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals