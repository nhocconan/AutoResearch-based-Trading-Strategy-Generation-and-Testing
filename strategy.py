#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA200 filter.
# 4h Donchian provides trend direction from higher timeframe structure.
# Volume confirmation ensures breakout has participation.
# 1d EMA200 filter avoids counter-trend trades in strong trends.
# Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year to minimize fee drag.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).
name = "1h_Donchian40_4hVolume_1dEMA200"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data for EMA200 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 40-period Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=40, min_periods=40).max().values
    donchian_low = pd.Series(low_4h).rolling(window=40, min_periods=40).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 200-period EMA on 1d
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume and above 1d EMA200
            if vol_confirm and close[i] > donchian_high_aligned[i] and close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume and below 1d EMA200
            elif vol_confirm and close[i] < donchian_low_aligned[i] and close[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals