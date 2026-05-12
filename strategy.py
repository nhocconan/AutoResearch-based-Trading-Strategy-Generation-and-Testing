#!/usr/bin/env python3
"""
1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_1DVOLUME
Hypothesis: For 1h timeframe, use 4h trend (EMA50) and 1d volume spike to filter Camarilla R1/S1 breakouts.
Only take breakouts aligned with 4h trend and confirmed by above-average 1d volume to reduce false signals.
Designed for ~15-35 trades/year on 1h to minimize fee drag while capturing institutional moves in bull/bear markets.
"""
name = "1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_1DVOLUME"
timeframe = "1h"
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
    
    # Calculate Camarilla levels from previous day
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        camarilla_r1[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 4
        camarilla_s1[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 4
    
    # 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d  # Current volume / 20-day average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike and 4h uptrend
            if (high[i] > camarilla_r1[i] and 
                vol_spike_aligned[i] > 1.5 and
                close[i] > ema_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with volume spike and 4h downtrend
            elif (low[i] < camarilla_s1[i] and 
                  vol_spike_aligned[i] > 1.5 and
                  close[i] < ema_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (reversion to mean)
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (reversion to mean)
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals