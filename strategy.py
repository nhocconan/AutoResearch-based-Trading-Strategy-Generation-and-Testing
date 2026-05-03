#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Weekly pivot levels provide institutional reference points for trend direction
# Donchian breakout captures momentum with weekly context to avoid counter-trend trades
# Volume confirmation (>2.0x 6h volume EMA) filters false breakouts
# Works in bull/bear markets by only taking breakouts in direction of weekly trend
# Target: 75-150 total trades over 4 years (19-37/year) for optimal fee balance

name = "6h_Donchian20_1wPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot levels: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2.0 * pp_1w - low_1w
    s1_1w = 2.0 * pp_1w - high_1w
    weekly_trend = close_1w > pp_1w  # 1 for bullish, 0 for bearish
    
    # Align weekly data to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Calculate Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start from lookback to have valid Donchian
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Donchian breakout signals with weekly pivot filter
        # Long: Break above Donchian high + weekly bullish trend + volume spike
        # Short: Break below Donchian low + weekly bearish trend + volume spike
        if position == 0:
            if (high[i] > highest_high[i] and 
                weekly_trend_aligned[i] > 0.5 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            elif (low[i] < lowest_low[i] and 
                  weekly_trend_aligned[i] < 0.5 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low OR weekly trend turns bearish
            if low[i] < lowest_low[i] or weekly_trend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high OR weekly trend turns bullish
            if high[i] > highest_high[i] or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals