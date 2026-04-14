#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Spike and 1w EMA Trend Filter
# Takes long when price breaks above 4h Donchian upper band (20) with 1d volume spike (>2x 20-period avg) and price > 1w EMA50
# Takes short when price breaks below 4h Donchian lower band (20) with 1d volume spike (>2x 20-period avg) and price < 1w EMA50
# Exits when price crosses back below/above the 4h Donchian midline or volume drops below average
# Designed to capture strong trends with volume confirmation in both bull and bear markets
# Target: 20-50 trades per symbol over 4 years (5-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i]  # Use current 4h volume as proxy for 1d volume spike detection
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike and bullish trend (price > 1w EMA50)
            if (price > donchian_high_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                price > ema50_1w_aligned[i]):                   # Bullish trend filter
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with volume spike and bearish trend (price < 1w EMA50)
            elif (price < donchian_low_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                  price < ema50_1w_aligned[i]):                   # Bearish trend filter
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian mid or volume drops below average
            if price < donchian_mid_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian mid or volume drops below average
            if price > donchian_mid_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dVolumeSpike_1wEMA50"
timeframe = "4h"
leverage = 1.0