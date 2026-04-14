#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Confirmation and 1w Trend Filter
# Takes long when price breaks above 4h Donchian upper band with 1d volume spike and 1w EMA200 above price
# Takes short when price breaks below 4h Donchian lower band with 1d volume spike and 1w EMA200 below price
# Exits when price crosses back below/above the 4h Donchian midline
# Designed to capture strong trends with volume confirmation in trending markets, avoiding counter-trend trades
# Target: 20-50 trades per symbol over 4 years (5-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h, 1d, and 1w data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike and bullish weekly trend
            if (price > donchian_high_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                ema200_1w_aligned[i] < price):                  # Weekly uptrend (price above EMA200)
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with volume spike and bearish weekly trend
            elif (price < donchian_low_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  ema200_1w_aligned[i] > price):                  # Weekly downtrend (price below EMA200)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian mid
            if price < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian mid
            if price > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0