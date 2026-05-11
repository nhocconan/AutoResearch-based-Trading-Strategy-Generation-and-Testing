#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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
    
    # Calculate weekly trend using 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    weekly_ema = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = pd.Series(close_1w) > pd.Series(weekly_ema)
    weekly_trend_up = weekly_trend_up.values
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    
    # Calculate Camarilla levels from previous day
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    for i in range(1, n):
        pc = close[i-1]
        ph = high[i-1]
        pl = low[i-1]
        camarilla_r1[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_s1[i] = pc - (ph - pl) * 1.1 / 12
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_up_aligned[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above Camarilla R1 + weekly uptrend + volume confirmation
            if close[i] > camarilla_r1[i] and weekly_trend_up_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below Camarilla S1 + weekly downtrend + volume confirmation
            elif close[i] < camarilla_s1[i] and not weekly_trend_up_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below Camarilla S1
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above Camarilla R1
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals