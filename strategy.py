#!/usr/bin/env python3
"""
4h_Donchian20_Volume_12hTrend
Hypothesis: Use 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
The 12h EMA defines the trend direction, Donchian breakouts capture momentum in that direction.
Volume filter avoids false breakouts. Designed for ~25-40 trades/year, works in bull/bear via trend filter.
"""

name = "4h_Donchian20_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already aligned since same timeframe)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA(50) and 4h Donchian (20)
    start_idx = 50  # for 12h EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: price vs 12h EMA50
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long: bullish 12h trend AND price breaks above 4h Donchian high with volume
            if bullish_trend and high[i] > donchian_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish 12h trend AND price breaks below 4h Donchian low with volume
            elif bearish_trend and low[i] < donchian_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low OR 12h trend turns bearish
            if low[i] < donchian_low_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high OR 12h trend turns bullish
            if high[i] > donchian_high_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals