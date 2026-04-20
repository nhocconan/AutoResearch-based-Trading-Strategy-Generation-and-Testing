#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Spike + Chop Regime Filter
# - Long when price breaks above Donchian(20) high with volume spike and chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low with volume spike and chop > 61.8 (range)
# - Exit when price reverses to opposite Donchian band or chop < 38.2 (trend)
# - Uses volume spike (volume > 1.5 * avg volume) to confirm breakouts
# - Chop filter prevents whipsaws in strong trends, improves performance in ranging markets
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 19-50 trades per year per symbol (75-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for chop calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Chop Index (14) on 12h timeframe
    atr_12h = []
    for i in range(len(close_12h)):
        if i == 0:
            tr = high_12h[i] - low_12h[i]
        else:
            tr = max(high_12h[i] - low_12h[i], 
                     abs(high_12h[i] - close_12h[i-1]), 
                     abs(low_12h[i] - close_12h[i-1]))
        atr_12h.append(tr)
    
    atr_12h = np.array(atr_12h)
    atr_sum_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_12h = chop.values
    
    # Align 12h Chop to 4h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate Donchian Channels (20) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    vol_4h = prices['volume'].values
    
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume (20) for volume spike detection
    avg_vol_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup period
        # Skip if NaN in indicators
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(avg_vol_20[i]) or np.isnan(chop_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = vol_4h[i]
        highest_high = highest_high_20[i]
        lowest_low = lowest_low_20[i]
        avg_vol = avg_vol_20[i]
        chop_val = chop_12h_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high + volume spike + chop > 61.8 (range)
            if price > highest_high and vol > 1.5 * avg_vol and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low + volume spike + chop > 61.8 (range)
            elif price < lowest_low and vol > 1.5 * avg_vol and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR chop < 38.2 (trend)
            if price < lowest_low or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR chop < 38.2 (trend)
            if price > highest_high or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0