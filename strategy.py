#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot shows bullish bias (price > weekly pivot) AND volume > 2x 20-period average
# Short when price breaks below Donchian(20) low AND weekly pivot shows bearish bias (price < weekly pivot) AND volume > 2x 20-period average
# Exit when price returns to Donchian(20) midpoint OR weekly pivot bias flips
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian breakouts capture institutional momentum, weekly pivot provides higher-timeframe regime filter to avoid counter-trend trades,
# volume spike confirms institutional participation. Works in both bull (longs in uptrend+breakouts) and bear (shorts in downtrend+breakouts) markets.

name = "6h_Donchian20_WeeklyPivot_Trend_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for pivot regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly bullish bias: price > weekly pivot
    weekly_bullish = pivot_1w  # we'll align this and compare close > aligned
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    
    # Calculate Donchian(20) channels on 6h data
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2x 20-period average (institutional participation)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high AND weekly bullish bias AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_bullish_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low AND weekly bearish bias (price < pivot) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_bullish_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR weekly bias turns bearish
            if (close[i] <= donchian_mid[i] or 
                close[i] <= weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR weekly bias turns bullish
            if (close[i] >= donchian_mid[i] or 
                close[i] >= weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals