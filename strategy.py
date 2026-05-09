#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with daily pivot-based breakout and volume confirmation
# Uses daily pivot points (R1/S1) with 1-week trend filter and volume spike
# Designed to capture breakouts in both bull and bear markets by requiring volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Edge: Pivot levels act as dynamic support/resistance, volume confirms institutional interest
name = "6h_1d_Pivot_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Get daily data for pivot points, trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for standard pivot point calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Standard pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Trend filter: 1-week EMA50 (using 1d data - 50 periods = ~10 weeks)
    # We use 50-day EMA as proxy for weekly trend to ensure sufficient data
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current day volume > 2.0 * 20-day average (significant spike)
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 2.0)
    
    # Align all to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_6h[i]
        s1_val = s1_6h[i]
        r2_val = r2_6h[i]
        s2_val = s2_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: break above R1 with volume spike and above weekly trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume spike and below weekly trend
            elif close[i] < s1_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below S1 (mean reversion) or reverse signal with volume
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            elif close[i] < s2_val and vol_filter:  # Strong break below S2 with volume
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above R1 (mean reversion) or reverse signal with volume
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            elif close[i] > r2_val and vol_filter:  # Strong break above R2 with volume
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals