#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week high/low breakouts with volume confirmation and 1-day trend filter
# Breakout above weekly high or below weekly low with volume > 1.8x 20-period average indicates strong momentum
# Trend filter: 34-period EMA on 1d timeframe to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, filter prevents false breakouts in chop
# Target: 75-200 total trades over 4 years (19-50/year) with 0.28 position sizing

name = "4h_WeeklyBreakout_VolumeTrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly high/low ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high and low for breakout levels
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Align weekly levels to 4h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_week_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # Calculate 1-day EMA34 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly high with volume confirmation and uptrend
            if close[i] > weekly_high_aligned[i] and volume_filter[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.28
                position = 1
            # Short breakout: price breaks below weekly low with volume confirmation and downtrend
            elif close[i] < weekly_low_aligned[i] and volume_filter[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly low or reverse signal with volume
            if close[i] < weekly_low_aligned[i] or (close[i] < ema_34_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price breaks above weekly high or reverse signal with volume
            if close[i] > weekly_high_aligned[i] or (close[i] > ema_34_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals