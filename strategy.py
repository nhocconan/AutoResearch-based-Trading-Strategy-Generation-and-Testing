#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND chop(14) > 61.8 (ranging).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND chop(14) > 61.8.
# Uses discrete position size 0.25. Donchian captures breakouts, volume confirms participation, chop filter avoids whipsaws in strong trends.
# Designed to work in both bull (breakouts up) and bear (breakdowns down) markets by being direction-agnostic.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Choppiness Index (14-period) ===
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / np.log(14) / (highest_high_14 - lowest_low_14))
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        chop_value = chop[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian low (breakdown) or chop drops below 38.2 (trending)
            if price < donchian_low or chop_value < 38.2:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian high (breakout) or chop drops below 38.2 (trending)
            if price > donchian_high or chop_value < 38.2:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND chop > 61.8 (ranging)
            if price > donchian_high and vol_spike and chop_value > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian low AND volume spike AND chop > 61.8 (ranging)
            elif price < donchian_low and vol_spike and chop_value > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0