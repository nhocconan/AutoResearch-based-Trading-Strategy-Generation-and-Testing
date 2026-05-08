#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels with volume confirmation and trend filter.
# Camarilla levels (S1, S2, S3, R1, R2, R3) are strong support/resistance zones.
# Long when price breaks above R1 with volume in uptrend; short when breaks below S1 with volume in downtrend.
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (20-50/year) to minimize fee drag and capture high-probability breakouts.

name = "4h_Camarilla_R1S1_Breakout_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    camarilla_r1 = np.zeros_like(close_1d)  # Resistance 1
    camarilla_s1 = np.zeros_like(close_1d)  # Support 1
    
    for i in range(1, len(close_1d)):
        # Previous day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point
        pivot = (ph + pl + pc) / 3.0
        
        # Range
        rng = ph - pl
        
        # Camarilla levels
        camarilla_r1[i] = pc + (rng * 1.1 / 12)  # R1
        camarilla_s1[i] = pc - (rng * 1.1 / 12)  # S1
    
    # First day has no prior data
    camarilla_r1[0] = camarilla_s1[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = ema_21_1w[1:] > ema_21_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with daily index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above R1 with volume in uptrend
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > camarilla_r1_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below S1 with volume in downtrend
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < camarilla_s1_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals