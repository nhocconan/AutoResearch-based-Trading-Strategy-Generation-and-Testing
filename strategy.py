#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 12h Donchian breakout with 1d trend filter
# Uses 12h Donchian(20) breakout for entry timing, confirmed by 1d EMA(50) trend
# Volume confirmation >1.5x 20-period average to filter weak breakouts
# Works in bull/bear: breaks capture trends, trend filter avoids counter-trend trades
# Target: 80-150 total trades over 4 years (20-37/year) with 0.28 position sizing

name = "6h_Donchian20_12h_1dTrend_VolumeFilter_v1"
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
    
    # Calculate 12h Donchian channels ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian(20) - upper and lower bands
    donch_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Calculate 1d EMA(50) trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 12h Donchian high with volume and uptrend
            if close[i] > donch_high_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.28
                position = 1
            # Short entry: price breaks below 12h Donchian low with volume and downtrend
            elif close[i] < donch_low_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low (failed breakout)
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high (failed breakdown)
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals