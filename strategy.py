#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Pivot Points (Classic) with volume confirmation and trend filter.
# Pivot points act as key support/resistance levels. Price tends to respect these levels.
# Long when price bounces from S1/S2 in uptrend with volume confirmation.
# Short when price is rejected from R1/R2 in downtrend with volume confirmation.
# Uses 1-week trend filter (EMA21 slope) to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (15-30/year) to minimize whipsaw and capture high-probability bounces.

name = "6h_PivotBounce_TrendFilter_Volume"
timeframe = "6h"
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
    
    # Calculate Classic Pivot Points: P = (H+L+C)/3
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)  # Resistance 1
    s1 = np.zeros_like(close_1d)  # Support 1
    r2 = np.zeros_like(close_1d)  # Resistance 2
    s2 = np.zeros_like(close_1d)  # Support 2
    
    for i in range(1, len(close_1d)):
        # Prior day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point and support/resistance levels
        p = (ph + pl + pc) / 3.0
        pivot[i] = p
        r1[i] = 2 * p - pl
        s1[i] = 2 * p - ph
        r2[i] = p + (ph - pl)
        s2[i] = p - (ph - pl)
    
    # First day has no prior data
    pivot[0] = r1[0] = s1[0] = r2[0] = s2[0] = np.nan
    
    # Align Pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
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
    
    # 6x EMA(50) for intermediate trend and dynamic support/resistance
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.8x 30-period EMA
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bounce from S1/S2 in uptrend with volume
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > ema_50[i] and             # Above intermediate EMA
                ((close[i] >= s1_aligned[i] * 0.995 and close[i] <= s1_aligned[i] * 1.005) or
                 (close[i] >= s2_aligned[i] * 0.995 and close[i] <= s2_aligned[i] * 1.005)) and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: rejection from R1/R2 in downtrend with volume
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < ema_50[i] and            # Below intermediate EMA
                  ((close[i] >= r1_aligned[i] * 0.995 and close[i] <= r1_aligned[i] * 1.005) or
                   (close[i] >= r2_aligned[i] * 0.995 and close[i] <= r2_aligned[i] * 1.005)) and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below pivot or trend turns down
            if close[i] < pivot_aligned[i] * 0.995 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above pivot or trend turns up
            if close[i] > pivot_aligned[i] * 1.005 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals