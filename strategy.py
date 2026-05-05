#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND weekly pivot trend is up AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20-period) AND weekly pivot trend is down AND volume > 1.5x 20-period average
# Exit when price crosses back to Donchian midpoint OR weekly pivot trend flips
# Weekly pivot provides higher-timeframe structure to avoid counter-trend trades in choppy markets
# Donchian breakout captures momentum, volume filter ensures institutional participation
# Target: 12-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends

name = "6h_Donchian20_WeeklyPivot_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # Support/resistance: R1 = 2*P - L, S1 = 2*P - H
    prev_high = np.concatenate([[np.nan], df_1w['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1w['low'].values[:-1]])
    prev_close = np.concatenate([[np.nan], df_1w['close'].values[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Weekly trend: price above pivot = uptrend, below = downtrend
    weekly_uptrend = prev_close > pivot
    weekly_downtrend = prev_close < pivot
    
    # Align weekly data to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Donchian channels (20-period) on 6h data
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND weekly uptrend AND volume spike
            if (close[i] > donchian_high[i] and 
                weekly_uptrend_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND weekly downtrend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_downtrend_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian midpoint OR weekly trend flips to downtrend
            if (close[i] < donchian_mid[i] or 
                weekly_downtrend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian midpoint OR weekly trend flips to uptrend
            if (close[i] > donchian_mid[i] or 
                weekly_uptrend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals