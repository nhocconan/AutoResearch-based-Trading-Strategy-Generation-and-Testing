#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla pivot levels from previous day (use previous day's data)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # We use previous day's values, so shift by 1
    if len(close_1d) > 1:
        H_prev = np.roll(high_1d, 1)
        L_prev = np.roll(low_1d, 1)
        C_prev = np.roll(close_1d, 1)
        H_prev[0] = np.nan
        L_prev[0] = np.nan
        C_prev[0] = np.nan
        
        pivot = (H_prev + L_prev + C_prev) / 3.0
        R1 = C_prev + (H_prev - L_prev) * 1.1 / 12.0
        S1 = C_prev - (H_prev - L_prev) * 1.1 / 12.0
    else:
        pivot = np.full_like(close_1d, np.nan)
        R1 = np.full_like(close_1d, np.nan)
        S1 = np.full_like(close_1d, np.nan)
    
    # Align Camarilla levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike on 4h: current volume > 2.0 x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(ema34_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + price above daily EMA34 + volume spike
            if (close[i] > R1_4h[i] and 
                close[i] > ema34_4h[i] and 
                vol_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + price below daily EMA34 + volume spike
            elif (close[i] < S1_4h[i] and 
                  close[i] < ema34_4h[i] and 
                  vol_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 or below daily EMA34
            if close[i] < S1_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above R1 or above daily EMA34
            if close[i] > R1_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals