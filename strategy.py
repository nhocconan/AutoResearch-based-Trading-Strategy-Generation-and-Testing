#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Donchian20_WeeklyTrend_VolumeSpike"
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
    
    # Get weekly data once for Donchian and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-week period)
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20w = high_20w
    donchian_low_20w = low_20w
    
    # Weekly trend: price above/below 20-week SMA
    sma_20w = pd.Series(df_1w['close'].values).rolling(window=20, min_periods=20).mean().values
    weekly_trend = df_1w['close'].values > sma_20w  # True for uptrend
    
    # Align weekly data to 6h timeframe
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Volume spike detection: current volume > 2.0 * 20-period average (on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20w_aligned[i]) or np.isnan(donchian_low_20w_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high with volume spike and weekly uptrend
            long_cond = (close[i] > donchian_high_20w_aligned[i] and vol_spike[i] and weekly_trend_aligned[i] > 0.5)
            
            # Short entry: price breaks below weekly Donchian low with volume spike and weekly downtrend
            short_cond = (close[i] < donchian_low_20w_aligned[i] and vol_spike[i] and weekly_trend_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low (reversal signal)
            if close[i] < donchian_low_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high (reversal signal)
            if close[i] > donchian_high_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with volume spike confirmation and weekly trend filter on 6h timeframe.
# Enters long when price breaks above 20-week Donchian high with volume spike and weekly uptrend (price > 20-week SMA).
# Enters short when price breaks below 20-week Donchian low with volume spike and weekly downtrend (price < 20-week SMA).
# Exits when price reverses back through the opposite Donchian band.
# Uses weekly timeframe for structure and trend, 6h for execution.
# Volume spike filters out low-momentum breakouts.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).
# Targets 15-25 trades/year on 6h timeframe (60-100 total over 4 years).
# Uses discrete sizing (0.25) to minimize churn from signal changes.