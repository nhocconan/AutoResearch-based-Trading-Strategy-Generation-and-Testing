#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channel (20-period) on 12h data
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donchian_high = high_12h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_12h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate EMA(50) on 12h close for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    volume_1d_series = pd.Series(volume_1d)
    avg_volume_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align average volume to primary timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 50-period EMA and 20-period Donchian
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 12h EMA50 with volume confirmation
            if price > donchian_high_aligned[i] and price > ema_50_12h_aligned[i] and vol > 1.5 * avg_volume_1d_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND below 12h EMA50 with volume confirmation
            elif price < donchian_low_aligned[i] and price < ema_50_12h_aligned[i] and vol > 1.5 * avg_volume_1d_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if price < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if price > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_EMA_Volume"
timeframe = "12h"
leverage = 1.0