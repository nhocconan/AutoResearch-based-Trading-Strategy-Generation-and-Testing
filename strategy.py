#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Pivot Point levels with volume confirmation and trend filter.
# Pivot points (R1/S1) act as strong support/resistance in trending markets.
# Long when price breaks above R1 with volume confirmation in daily uptrend.
# Short when price breaks below S1 with volume confirmation in daily downtrend.
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (15-30/year) to minimize whipsaw and capture high-probability breakouts.

name = "4h_PivotBreakout_TrendFilter_Volume"
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
    
    # Get daily data for Pivot Point calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot Points and support/resistance levels from prior day
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)  # Resistance 1
    s1 = np.zeros_like(close_1d)  # Support 1
    
    for i in range(1, len(close_1d)):
        # Prior day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot Point calculation
        pivot[i] = (ph + pl + pc) / 3.0
        # Resistance 1 and Support 1
        r1[i] = 2 * pivot[i] - pl
        s1[i] = 2 * pivot[i] - ph
    
    # First day has no prior data
    pivot[0] = r1[0] = s1[0] = np.nan
    
    # Align Pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
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
    
    # 4h EMA(50) for intermediate trend and dynamic support/resistance
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above R1 with volume in daily uptrend
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > ema_50[i] and             # Above intermediate EMA
                close[i] > r1_aligned[i] * 1.002 and  # Break above R1 (0.2% buffer)
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below S1 with volume in daily downtrend
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < ema_50[i] and            # Below intermediate EMA
                  close[i] < s1_aligned[i] * 0.998 and  # Break below S1 (0.2% buffer)
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below pivot or trend turns down
            if close[i] < pivot_aligned[i] * 0.998 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above pivot or trend turns up
            if close[i] > pivot_aligned[i] * 1.002 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals