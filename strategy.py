#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h Volume Spike + 1d Trend Filter (EMA50)
# - Long when price breaks above Donchian upper band (20) on 4h + volume spike (12h) + price > EMA50 (1d)
# - Short when price breaks below Donchian lower band (20) on 4h + volume spike (12h) + price < EMA50 (1d)
# - Exit when price crosses back through Donchian middle (10-period average of bands)
# - Volume spike defined as current volume > 1.5 * 20-period average volume on 12h
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for volume spike calculation
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h
    avg_vol_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = (vol_12h > 1.5 * avg_vol_20_12h).astype(float)
    
    # Align 12h volume spike to 4h timeframe
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Donchian/EMA warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(vol_spike_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = vol_spike_12h_aligned[i]
        ema50 = ema_50_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        
        if position == 0:
            # Long entry: price breaks above upper band + volume spike + price > EMA50
            if price > upper and vol_spike > 0.5 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + volume spike + price < EMA50
            elif price < lower and vol_spike > 0.5 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below middle band
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above middle band
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_EMA50Filter"
timeframe = "4h"
leverage = 1.0