#!/usr/bin/env python3
# 12h_weekly_donchian_volume_chop_v1
# Hypothesis: 12h strategy using 1w Donchian channel breakouts with volume confirmation and chop regime filter.
# Long: price breaks above 20-period 1w Donchian high with volume > 1.5x average volume AND market is trending (CHOP < 61.8)
# Short: price breaks below 20-period 1w Donchian low with volume > 1.5x average volume AND market is trending (CHOP < 61.8)
# Exit: price reverses to midpoint of Donchian channel or regime shifts to choppy.
# Designed to capture medium-term breakouts from weekly structure while avoiding false breakouts in ranging markets.
# Weekly Donchian provides institutional reference points; volume confirms participation; chop filter avoids whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period high and low
    high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to LTF
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate volume ratio (current vs 50-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(50, n):
        vol_sma[i] = np.mean(volume[i-50:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Calculate Chopiness Index for regime filter (14-period)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        atr_sum = 0
        for j in range(i-13, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        atr = atr_sum / 14
        max_high = np.max(high[i-13:i+1])
        min_low = np.min(low[i-13:i+1])
        if max_high != min_low:
            chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        vol_r = vol_ratio[i]
        ch = chop[i]
        price = close[i]
        
        if np.isnan(vol_r) or np.isnan(ch):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        
        if np.isnan(upper) or np.isnan(lower):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < (upper + lower) / 2 or vol_r < 1.3 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > (upper + lower) / 2 or vol_r < 1.3 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > upper and vol_r > 1.5 and ch < 61.8:
                position = 1
                signals[i] = 0.25
            elif price < lower and vol_r > 1.5 and ch < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals