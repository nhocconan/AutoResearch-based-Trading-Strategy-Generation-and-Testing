#!/usr/bin/env python3
# 12h Camarilla R1/S1 Breakout + Volume Spike + Weekly Trend Filter
# Hypothesis: Camarilla pivot levels (R1/S1) from weekly data provide strong support/resistance.
# Breaking these levels with volume confirmation and aligned with weekly trend (price > weekly EMA34)
# captures strong moves in both bull and bear markets. Designed for low trade frequency (~15-30/year)
# to minimize fee decay on 12h timeframe.
name = "12h_Camarilla_R1S1_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
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
    
    # === Weekly Data for Camarilla Levels and Trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from weekly data (previous week's HLC)
    weekly_high_1w = df_1w['high'].values
    weekly_low_1w = df_1w['low'].values
    weekly_close_1w = df_1w['close'].values
    
    rng = weekly_high_1w - weekly_low_1w
    R1 = weekly_close_1w + rng * 1.1 / 12  # R1 = C + (H-L)*1.1/12
    S1 = weekly_close_1w - rng * 1.1 / 12  # S1 = C - (H-L)*1.1/12
    
    # Shift to get previous week's levels (this week's R1/S1 based on last week's HLC)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    
    # Align to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1w, R1_prev)
    S1_12h = align_htf_to_ltf(prices, df_1w, S1_prev)
    
    # === Weekly EMA34 for trend filter ===
    ema_34_1w = pd.Series(weekly_close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume Spike (20-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above weekly EMA34 (uptrend)
            if (close[i] > R1_12h[i] and 
                vol_spike[i] and
                close[i] > ema_34_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below weekly EMA34 (downtrend)
            elif (close[i] < S1_12h[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (reversal)
            if close[i] < S1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (reversal)
            if close[i] > R1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals