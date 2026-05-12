#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + Volume Spike + Daily Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3) derived from daily data act as strong support/resistance.
Breaking these levels with volume confirmation and aligned with daily trend (price > EMA34) captures
strong moves in both bull and bear markets. Designed for low trade frequency (~25/year) to minimize fee decay.
"""
name = "4h_Camarilla_R3S3_Breakout_DailyTrend_Volume"
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
    
    # === Daily Data for Camarilla Levels and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily data (previous day's HLC)
    daily_high_1d = df_1d['high'].values
    daily_low_1d = df_1d['low'].values
    daily_close_1d = df_1d['close'].values
    
    rng = daily_high_1d - daily_low_1d
    R3 = daily_close_1d + rng * 1.1 / 4  # R3 = C + (H-L)*1.1/4
    S3 = daily_close_1d - rng * 1.1 / 4  # S3 = C - (H-L)*1.1/4
    
    # Shift to get previous day's levels (today's R3/S3 based on yesterday's HLC)
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    R3_prev[0] = np.nan
    S3_prev[0] = np.nan
    
    # Align to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3_prev)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3_prev)
    
    # === Daily EMA34 for trend filter ===
    ema_34_1d = pd.Series(daily_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > R3_4h[i] and 
                vol_spike[i] and
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < S3_4h[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal)
            if close[i] < S3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal)
            if close[i] > R3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals