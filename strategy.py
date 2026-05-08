#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day pivot points with volume confirmation and trend filter.
# Pivot points (S1, R1) act as strong support/resistance in trending markets.
# Long when price bounces from S1 in uptrend with volume confirmation.
# Short when price rejects R1 in downtrend with volume confirmation.
# Uses 1-day trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (12-37/year) to minimize fee drag and capture high-probability reversals.

name = "12h_Pivot_Bounce_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot point calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and support/resistance levels from prior day
    pivot = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)  # Support 1
    r1 = np.zeros_like(close_1d)  # Resistance 1
    
    for i in range(1, len(close_1d)):
        # Prior day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point calculation
        pivot[i] = (ph + pl + pc) / 3.0
        
        # Support and resistance levels
        s1[i] = 2 * pivot[i] - ph
        r1[i] = 2 * pivot[i] - pl
    
    # First day has no prior data
    pivot[0] = s1[0] = r1[0] = np.nan
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Get daily trend filter using EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising daily EMA
    daily_trend_up = np.concatenate([[False], daily_trend_up])  # Align with daily index
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    
    # Volume confirmation: current volume > 2.0x 24-period EMA
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(daily_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bounce from S1 in uptrend with volume confirmation
            if (daily_trend_aligned[i] > 0.5 and  # Daily uptrend
                close[i] >= s1_aligned[i] * 0.995 and  # At or above S1 (allow 0.5% slack)
                close[i] <= s1_aligned[i] * 1.005 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: rejection at R1 in downtrend with volume confirmation
            elif (daily_trend_aligned[i] <= 0.5 and  # Daily downtrend
                  close[i] <= r1_aligned[i] * 1.005 and  # At or below R1
                  close[i] >= r1_aligned[i] * 0.995 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S1 or trend turns down
            if close[i] < s1_aligned[i] * 0.995 or daily_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R1 or trend turns up
            if close[i] > r1_aligned[i] * 1.005 or daily_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals