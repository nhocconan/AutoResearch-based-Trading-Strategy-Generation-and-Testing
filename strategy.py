#!/usr/bin/env python3
"""
6H Weekly Pivot Mean Reversion with Volume Spike and 1d Trend Filter
Strategy: Fade extreme weekly pivot levels (R4/S4) with volume spike confirmation
Only trade when 1d EMA trend aligns with mean reversion direction
Exit when price reverts to weekly pivot point (PP)
Designed to work in both trending and ranging markets by fading extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_meanrev_volume_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (calculated from weekly OHLC) ===
    # Get weekly data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate pivot points from weekly OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*PP - L, S1 = 2*PP - H
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    # R2 = PP + (H - L), S2 = PP - (H - L)
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    # R3 = H + 2*(PP - L), S3 = L - 2*(H - PP)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    r4 = r3 + (weekly_high - weekly_low)
    s4 = s3 - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe (with look-ahead prevention)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods = 6 days
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to or above pivot point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to or below pivot point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Require volume spike (at least 1.5x average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions: Fade extreme levels with trend alignment
            # Long when price touches/falls below S4 AND 1d trend is up (mean reversion in uptrend)
            if close[i] <= s4_aligned[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                position = 1
                signals[i] = 0.25
            # Short when price touches/rises above R4 AND 1d trend is down (mean reversion in downtrend)
            elif close[i] >= r4_aligned[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals