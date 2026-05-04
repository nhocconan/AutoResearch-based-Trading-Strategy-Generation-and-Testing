#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Donchian(20) provides clear breakout levels from recent price extremes.
# Weekly pivot (R1/S1) gives higher-timeframe directional bias to avoid counter-trend trades.
# Volume spike (>1.8 x 20-period EMA) ensures institutional participation and reduces whipsaws.
# Designed for 6h timeframe targeting 75-150 total trades over 4 years (19-38/year).
# Works in bull markets via trend-aligned breakouts and in bear markets via filtered mean-reversion at extreme levels.

name = "6h_Donchian20_1wPivot_Direction_VolumeSpike"
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
    
    # Get 1d and 1w data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for short-term trend filter (not HTF, so calculate directly)
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly pivot points (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1 and S1 levels
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_20[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(pp_1w_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Short-term trend: bullish if close > EMA20, bearish if close < EMA20
        bullish_trend = close[i] > ema_20[i]
        bearish_trend = close[i] < ema_20[i]
        
        if position == 0:
            # Long: Close breaks above Donchian(20) upper + volume spike + price above weekly PP
            if (close[i] > high_max_20[i] and volume_spike and close[i] > pp_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian(20) lower + volume spike + price below weekly PP
            elif (close[i] < low_min_20[i] and volume_spike and close[i] < pp_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below Donchian(20) lower OR price crosses below weekly PP
            if (close[i] < low_min_20[i] or close[i] < pp_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above Donchian(20) upper OR price crosses above weekly PP
            if (close[i] > high_max_20[i] or close[i] > pp_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals