#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Daily price breaking above weekly Camarilla R1 in weekly uptrend or below weekly S1 in weekly downtrend
# continues with momentum. Volume confirmation filters false breakouts. Weekly trend uses 40-period EMA.
# Works in bull markets by following uptrends and bear markets by following downtrends.

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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend filter
    ema_40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate weekly Camarilla levels (standard formula)
    # Typical Price = (H + L + C) / 3
    # Range = H - L
    # R1 = Close + 1.1 * Range / 12
    # S1 = Close - 1.1 * Range / 12
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    typical_price = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    camarilla_r1 = weekly_close + 1.1 * weekly_range / 12
    camarilla_s1 = weekly_close - 1.1 * weekly_range / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume confirmation (20-period MA on daily = ~1 month)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA40 (40), weekly Camarilla (1), volume MA (20)
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = ema_40_1w_aligned[i] > ema_40_1w_aligned[i-1]  # rising EMA
        weekly_downtrend = ema_40_1w_aligned[i] < ema_40_1w_aligned[i-1]  # falling EMA
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: weekly uptrend + price breaks above Camarilla R1 + volume
            if weekly_uptrend and close[i] > camarilla_r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price breaks below Camarilla S1 + volume
            elif weekly_downtrend and close[i] < camarilla_s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down or price re-enters below R1
            if not weekly_uptrend or close[i] < camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up or price re-enters above S1
            if not weekly_downtrend or close[i] > camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals