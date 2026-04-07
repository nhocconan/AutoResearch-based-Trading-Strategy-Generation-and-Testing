#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with daily pivot direction and volume confirmation
# Uses Donchian(20) breakout for trend continuation, daily pivot points for directional bias,
# and volume surge confirmation to filter false breakouts. Designed for low trade frequency
# (target: 12-37 trades/year) with discrete position sizing to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at pivot levels.

name = "6h_donchian20_daily_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian(20) channel
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Daily pivot points (using previous day's OHLC)
    prev_open = df_1d['open'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Donchian breakout above upper band with volume surge
        # AND price above daily pivot (bullish bias)
        if (close[i] > donchian_high[i] and 
            volume_surge[i] and 
            close[i] > pivot_aligned[i]):
            signals[i] = 0.25  # 25% position
        
        # Short conditions: Donchian breakdown below lower band with volume surge
        # AND price below daily pivot (bearish bias)
        elif (close[i] < donchian_low[i] and 
              volume_surge[i] and 
              close[i] < pivot_aligned[i]):
            signals[i] = -0.25  # 25% short position
        
        # Exit conditions: price crosses mid-band or reverses at pivot levels
        elif (abs(close[i] - donchian_mid[i]) < 0.001 * close[i]):  # near mid-band
            signals[i] = 0.0
        elif (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]):  # resistance touch
            signals[i] = 0.0
        elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]):  # support touch
            signals[i] = 0.0
    
    return signals