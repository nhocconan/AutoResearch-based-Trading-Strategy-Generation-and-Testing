#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 3-day breakout with 1w EMA200 trend filter and volume confirmation
# Uses price breaking above/below the highest/lowest close of the past 3 days for entry
# Requires price to be above/below 1w EMA200 for trend direction filter
# Volume confirmation (>1.5x 20-bar average) ensures institutional participation
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Avoids choppy markets via EMA200 filter and volume requirement

name = "1d_3DayBreakout_1wEMA200_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 3-day highest/lowest close for breakout levels
    # Using 3-day lookback (3 daily bars)
    high_3d = pd.Series(close).rolling(window=3, min_periods=3).max().shift(1).values
    low_3d = pd.Series(close).rolling(window=3, min_periods=3).min().shift(1).values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_3d[i]) or np.isnan(low_3d[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 3-day high AND above 1w EMA200 AND volume confirmation
            if (close[i] > high_3d[i] and close[i] > ema_200_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 3-day low AND below 1w EMA200 AND volume confirmation
            elif (close[i] < low_3d[i] and close[i] < ema_200_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below 3-day low
            if close[i] < low_3d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above 3-day high
            if close[i] > high_3d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals