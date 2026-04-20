#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation
# - Uses 4h Donchian(20) for trend direction: long when price > 4h upper band, short when < 4h lower band
# - Entry on 1h when price breaks 20-period Donchian channel in direction of 4h trend
# - Requires volume > 1.5x 20-period average for confirmation
# - Designed for 1h timeframe with selective entries to avoid overtrading
# - Target: 15-37 trades per year per symbol (60-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend determination
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) for trend
    highest_high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = highest_high_20_4h
    donchian_lower_4h = lowest_low_20_4h
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Calculate 1h Donchian(20) for entry signals
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    highest_high_20_1h = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    lowest_low_20_1h = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    donchian_upper_1h = highest_high_20_1h
    donchian_lower_1h = lowest_low_20_1h
    
    # Calculate volume filter: volume > 1.5x 20-period average
    volume = prices['volume'].values
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * avg_volume_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or
            np.isnan(volume[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        vol = volume[i]
        vol_ma = avg_volume_20[i]
        upper_4h = donchian_upper_4h_aligned[i]
        lower_4h = donchian_lower_4h_aligned[i]
        upper_1h = donchian_upper_1h[i]
        lower_1h = donchian_lower_1h[i]
        
        if position == 0:
            # Long entry: price > 4h upper band AND price breaks 1h upper band AND volume confirmation
            if price > upper_4h and price > upper_1h and vol > vol_ma:
                signals[i] = 0.20
                position = 1
            # Short entry: price < 4h lower band AND price breaks 1h lower band AND volume confirmation
            elif price < lower_4h and price < lower_1h and vol > vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h lower band or volume drops significantly
            if price < lower_4h or vol < 0.5 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 4h upper band or volume drops significantly
            if price > upper_4h or vol < 0.5 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0