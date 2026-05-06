#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points for breakout direction and daily volume for confirmation
# Weekly pivot points (from prior week) establish major support/resistance levels
# Breakout above weekly R1 or below weekly S1 with daily volume > 1.5x 50-period average indicates strong momentum
# Trend filter: 100-period EMA on 6h timeframe to align with longer-term direction
# Works in bull/bear markets: breakouts capture new trends, reversals at weekly S2/R2 capture pullbacks
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_VolumeTrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Weekly Support and Resistance levels
    r1 = pivot + (range_ * 1.0)
    r2 = pivot + (range_ * 2.0)
    s1 = pivot - (range_ * 1.0)
    s2 = pivot - (range_ * 2.0)
    
    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily volume confirmation: >1.5x 50-period average (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Convert 6h volume to daily equivalent for comparison
    # We'll use the daily volume data directly for confirmation
    daily_volume = df_1d['volume'].values
    vol_ma_50 = pd.Series(daily_volume).rolling(window=50, min_periods=50).mean().values
    volume_factor = daily_volume > (1.5 * vol_ma_50)
    volume_aligned = align_htf_to_ltf(prices, df_1d, volume_factor)
    
    # Trend filter: 100-period EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_100 = close_series.ewm(span=100, adjust=False, min_periods=100).mean().values
    uptrend = close > ema_100
    downtrend = close < ema_100
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_aligned[i]) or np.isnan(ema_100[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with volume confirmation and uptrend
            if close[i] > r1_aligned[i] and volume_aligned[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly S1 with volume confirmation and downtrend
            elif close[i] < s1_aligned[i] and volume_aligned[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects weekly S2 with volume confirmation (bounce from support)
            elif close[i] < s2_aligned[i] and close[i] > s2_aligned[i] * 0.995 and volume_aligned[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects weekly R2 with volume confirmation (rejection from resistance)
            elif close[i] > r2_aligned[i] and close[i] < r2_aligned[i] * 1.005 and volume_aligned[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 (failed support) or reaches weekly R2 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 (failed resistance) or reaches weekly S2 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals