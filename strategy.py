#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 4h EMA50 trend filter and volume spike confirmation
# In trending markets (price > 4h EMA50): Donchian(20) breakout with volume > 2.0x 20-period EMA
# In ranging markets (price <= 4h EMA50): Donchian(20) breakout with volume > 3.0x 20-period EMA
# Uses discrete sizing (0.25) to minimize fees. Designed for 12h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# BTC/ETH edge: Donchian captures structure, EMA50 filters trend/regime, volume spike confirms institutional participation.

name = "12h_Donchian20_4hEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime: trending (close > 4h EMA50) or ranging (close <= 4h EMA50)
        is_trending = close[i] > ema_50_aligned[i]
        
        # Volume threshold: higher in ranging markets to avoid false breakouts
        vol_threshold = 3.0 if not is_trending else 2.0
        volume_confirm = volume[i] > (vol_threshold * vol_ema_20[i])
        
        if position == 0:
            # Look for Donchian breakout with volume confirmation
            if high[i] > highest_high[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif low[i] < lowest_low[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests Donchian middle OR volume drops significantly
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_mid or volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests Donchian middle OR volume drops significantly
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_mid or volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals