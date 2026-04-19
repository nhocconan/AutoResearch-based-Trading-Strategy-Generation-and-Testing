#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Donchian breakout (20-period), volume confirmation, and 1w EMA trend filter.
# Uses 1d Donchian channels to identify breakout levels, confirmed by volume spike.
# 1w EMA filter ensures trades align with higher timeframe trend (bull/bear agnostic).
# Target: 50-150 total trades over 4 years (12-37/year).
name = "4h_1d_Donchian20_1wEMA_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for EMA trend filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period high/low)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA (34-period) for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Trend filter: 1 = bullish trend (price > EMA), -1 = bearish trend (price < EMA)
        trend = 1 if close[i] > ema_1w_aligned[i] else -1
        
        if position == 0:
            # Long when price breaks above 1d Donchian high with volume and bullish trend
            if close[i] > donch_high_aligned[i] and volume_filter[i] and trend == 1:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 1d Donchian low with volume and bearish trend
            elif close[i] < donch_low_aligned[i] and volume_filter[i] and trend == -1:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below 1d Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above 1d Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals