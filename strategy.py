#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Use Camarilla pivot levels (R1/S1) from daily as breakout levels with 12h EMA trend filter and volume spike.
# Long when price breaks above R1 with price > 12h EMA and volume > 1.5x 20-period MA.
# Short when price breaks below S1 with price < 12h EMA and volume > 1.5x 20-period MA.
# Exit when price reverses back to the Camarilla pivot point (P).
# Designed to capture institutional breakout attempts with trend and volume confirmation.
# Targets 25-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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
    
    # Calculate daily Camarilla pivot levels (R1, S1, P)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla calculations
    P = (daily_high + daily_low + daily_close) / 3
    R1 = P + (daily_high - daily_low) * 1.1 / 12
    S1 = P - (daily_high - daily_low) * 1.1 / 12
    
    # Align to 4h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(P_aligned[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with price > 12h EMA and volume > 1.5x MA
            if close[i] > R1_aligned[i] and close[i] > ema_12h_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with price < 12h EMA and volume > 1.5x MA
            elif close[i] < S1_aligned[i] and close[i] < ema_12h_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back to pivot point P
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back to pivot point P
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals