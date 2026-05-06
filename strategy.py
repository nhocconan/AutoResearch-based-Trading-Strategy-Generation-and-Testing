#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation
# Uses weekly EMA200 for primary trend alignment to avoid counter-trend trades in both bull/bear markets
# Donchian breakout captures momentum bursts, volume filter ensures participation
# Designed for 6h timeframe with target of 60-120 total trades over 4 years (15-30/year)
# Discrete sizing 0.25 to manage risk and minimize fee churn

name = "6h_Donchian20_WeeklyEMA200_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 200 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA200 trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate 6h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_stop = 0.0
    short_stop = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND above weekly EMA200 AND volume filter
            if close[i] > high_rolling_max[i] and close[i] > ema200_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_stop = low_rolling_min[i]  # Initial stop at Donchian low
            # Short breakdown: price < Donchian low AND below weekly EMA200 AND volume filter
            elif close[i] < low_rolling_min[i] and close[i] < ema200_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_stop = high_rolling_max[i]  # Initial stop at Donchian high
        elif position == 1:
            # Trail stop: raise stop to Donchian low if price moves favorably
            long_stop = max(long_stop, low_rolling_min[i])
            # Exit if price hits trailing stop
            if close[i] <= long_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Trail stop: lower stop to Donchian high if price moves favorably
            short_stop = min(short_stop, high_rolling_max[i])
            # Exit if price hits trailing stop
            if close[i] >= short_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals