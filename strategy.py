#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and aligned daily trend (EMA34)
# capture momentum moves. Works in bull (buying R1 breakouts) and bear (selling S1 breakdowns).
# Uses 4h timeframe for entries, 1d for pivot levels and trend filter.
# Target: 20-50 trades/year to avoid fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We shift by 1 to use previous day's data (available at close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for previous day
    # Only calculate when we have a full previous day
    H_L = high_1d - low_1d
    R1 = close_1d + H_L * 1.1 / 12
    S1 = close_1d - H_L * 1.1 / 12
    
    # Shift to get previous day's levels (available at next bar)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R1_prev[0] = np.nan  # First value has no previous day
    S1_prev[0] = np.nan
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1_prev)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1_prev)
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_4h[i]) or 
            np.isnan(S1_4h[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + break above R1 + volume
            if uptrend and close[i] > R1_4h[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + break below S1 + volume
            elif downtrend and close[i] < S1_4h[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < R1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > S1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals