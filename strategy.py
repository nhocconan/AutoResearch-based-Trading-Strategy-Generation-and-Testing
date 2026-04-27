#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (CI) regime filter + Donchian(20) breakout + volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean reversion), < 38.2 indicates trending
# In trending regime (CI < 38.2): buy Donchian(20) breakout, sell breakdown
# In ranging regime (CI > 61.8): sell near upper band, buy near lower band (mean reversion)
# Volume confirmation reduces false signals. Designed to work in both bull and bear markets
# by adapting to market regime. Target: 20-50 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and Choppiness (same timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(19, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to 4h timeframe (no delay needed as calculated on same TF)
    dh_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate Choppiness Index (14-period)
    # CI = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(14)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full(len(df_4h), np.nan)
    for i in range(13, len(df_4h)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    choppiness = np.full(len(df_4h), np.nan)
    for i in range(13, len(df_4h)):
        atr_sum = np.sum(atr_14[i-13:i+1])
        max_high = np.max(high_4h[i-13:i+1])
        min_low = np.min(low_4h[i-13:i+1])
        if max_high > min_low and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            choppiness[i] = 50  # neutral when range is zero
    
    # Align Choppiness to 4h timeframe
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, choppiness)
    
    # Get 1d data for volume confirmation (higher timeframe volume filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 1d volume moving average (20-period)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align 1d volume MA to 4h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), ATR (14), volume MA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dh_4h_aligned[i]) or np.isnan(dl_4h_aligned[i]) or 
            np.isnan(chop_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20_1d_aligned[i]
        chop = chop_4h_aligned[i]
        
        # Volume filter: volume above average
        vol_filter = vol_now > vol_avg
        
        # Regime filters
        ranging = chop > 61.8   # ranging market (mean reversion)
        trending = chop < 38.2  # trending market (trend follow)
        neutral = ~(ranging | trending)  # neutral zone
        
        if position == 0:
            if trending and vol_filter:
                # Trending market: follow breakouts
                if price > dh_4h_aligned[i]:
                    signals[i] = size
                    position = 1
                elif price < dl_4h_aligned[i]:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif ranging and vol_filter:
                # Ranging market: mean reversion
                if price >= dh_4h_aligned[i]:
                    signals[i] = -size  # sell at upper band
                    position = -1
                elif price <= dl_4h_aligned[i]:
                    signals[i] = size   # buy at lower band
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # neutral zone or no volume
        elif position == 1:
            # Exit long: reversal signals
            if ranging and price <= dl_4h_aligned[i]:  # mean reversion exit
                signals[i] = 0.0
                position = 0
            elif trending and price < dl_4h_aligned[i]:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: reversal signals
            if ranging and price >= dh_4h_aligned[i]:  # mean reversion exit
                signals[i] = 0.0
                position = 0
            elif trending and price > dh_4h_aligned[i]:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Choppiness_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0