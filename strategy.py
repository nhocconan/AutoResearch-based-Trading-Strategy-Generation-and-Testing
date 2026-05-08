#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot levels with volume confirmation and trend filter.
# Weekly Camarilla levels (R1, R2, S1, S2) act as strong support/resistance.
# Long when price bounces from S1/S2 in uptrend with volume confirmation.
# Short when price rejects at R1/R2 in downtrend with volume confirmation.
# Uses weekly trend filter (EMA10) to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (7-25/year) to minimize fee drag.

name = "1d_CamarillaWeekly_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6
    r1_1w = pivot_1w + (range_1w * 1.1 / 12)
    r2_1w = pivot_1w + (range_1w * 1.1 / 6)
    s1_1w = pivot_1w - (range_1w * 1.1 / 12)
    s2_1w = pivot_1w - (range_1w * 1.1 / 6)
    
    # Align weekly Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Weekly trend filter: EMA(10) slope
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    weekly_trend_up = ema_10_1w[1:] > ema_10_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with weekly index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Daily volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bounce from S1/S2 in uptrend with volume
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                ((close[i] >= s1_aligned[i] * 0.995 and close[i] <= s1_aligned[i] * 1.005) or
                 (close[i] >= s2_aligned[i] * 0.995 and close[i] <= s2_aligned[i] * 1.005)) and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: rejection at R1/R2 in downtrend with volume
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  ((close[i] >= r1_aligned[i] * 0.995 and close[i] <= r1_aligned[i] * 1.005) or
                   (close[i] >= r2_aligned[i] * 0.995 and close[i] <= r2_aligned[i] * 1.005)) and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S2 or trend turns down
            if close[i] < s2_aligned[i] * 0.995 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R2 or trend turns up
            if close[i] > r2_aligned[i] * 1.005 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals