#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_chop_v3
# Hypothesis: Daily Donchian(20) breakouts with volume confirmation and weekly choppiness regime filter.
# Long: price breaks above Donchian(20) high with volume > 1.5x 20-period average AND weekly chop > 61.8 (range regime)
# Short: price breaks below Donchian(20) low with volume > 1.5x 20-period average AND weekly chop > 61.8 (range regime)
# Exit: price returns to Donchian(20) midpoint or opposite breakout level with volume confirmation
# Uses 1d primary timeframe with 1w HTF for choppiness regime filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and work in both bull/bear markets via mean reversion in ranging conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_chop_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly choppiness index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(1, len(df_1w)):
        tr = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
        if i < 14:
            atr_1w[i] = np.mean(atr_1w[1:i+1]) if np.any(~np.isnan(atr_1w[1:i+1])) else tr
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr) / 14
    
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_sum = np.sum(atr_1w[i-13:i+1])
        max_high = np.max(high_1w[i-13:i+1])
        min_low = np.min(low_1w[i-13:i+1])
        if max_high > min_low:
            chop_1w[i] = 100 * np.log10(atr_sum / np.log10(max_high - min_low)) / 14
        else:
            chop_1w[i] = 50.0
    
    # Align 1w choppiness to daily timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        chop = chop_1w_aligned[i]
        
        if np.isnan(vol_r) or np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or np.isnan(chop):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Range regime filter: only trade when weekly chop > 61.8 (ranging market)
        in_range_regime = chop > 61.8
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint or breaks below low with volume
            if price <= dm or (price < dl and vol_r > 1.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint or breaks above high with volume
            if price >= dm or (price > dh and vol_r > 1.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only enter in ranging regime
            if in_range_regime:
                # Long entry: price breaks above Donchian high with volume confirmation
                if price > dh and vol_r > 1.5:
                    position = 1
                    entry_price = price
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low with volume confirmation
                elif price < dl and vol_r > 1.5:
                    position = -1
                    entry_price = price
                    signals[i] = -0.25
    
    return signals