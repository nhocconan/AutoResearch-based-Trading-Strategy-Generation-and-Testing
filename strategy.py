#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 4h choppiness regime
# - Long when price breaks above Donchian(20) high + 1d volume > 2.0 * 10-day avg + CHOP(14) > 61.8 (range)
# - Short when price breaks below Donchian(20) low + 1d volume > 2.0 * 10-day avg + CHOP(14) > 61.8 (range)
# - Exit when price crosses back through Donchian(20) midpoint or volume drops below threshold
# - Uses 1d volume for institutional participation confirmation
# - Chop filter ensures trades occur in ranging markets where mean reversion works
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 2.0 * 10-day average volume
    vol_avg_10 = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_spike = vol_1d > (2.0 * vol_avg_10)
    vol_spike_1d = vol_spike.astype(float)  # 1.0 when spike, 0.0 otherwise
    
    # Align 1d volume spike to 4h timeframe
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate Choppiness Index (14-period) on 4h timeframe
    atr_14 = pd.Series(np.maximum(
        high_4h - low_4h,
        np.maximum(
            np.abs(high_4h - np.roll(close_4h, 1)),
            np.abs(low_4h - np.roll(close_4h, 1))
        )
    )).rolling(window=14, min_periods=14).mean().values
    
    # Handle first value for rolling calculations
    atr_14[0] = high_4h[0] - low_4h[0]
    
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(sum_atr_14 / np.log10(14)) / np.log10(range_14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # Fill NaN with neutral value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(chop[i]) or np.isnan(vol_spike_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol_spike = vol_spike_4h[i] > 0.5  # True if volume spike
        chop_value = chop[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + choppy market (range)
            if price > donchian_high[i] and vol_spike and chop_value > 61.8:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + choppy market (range)
            elif price < donchian_low[i] and vol_spike and chop_value > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid or volatility drops (chop < 38.2 = trending)
            if price < donchian_mid[i] or chop_value < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid or volatility drops (chop < 38.2 = trending)
            if price > donchian_mid[i] or chop_value < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0