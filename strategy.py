#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter combined with Donchian(20) breakout and volume confirmation
# Chop > 61.8 indicates range (mean revert), Chop < 38.2 indicates trend (follow breakout)
# Donchian breakout with volume filter captures momentum in trending regimes
# Works in bull/bear by adapting strategy based on market regime (trending vs ranging)
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First period
    
    # Calculate ATR(14) for 1d
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Choppiness Index (14)
    chop = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        sum_tr = np.sum(tr[i-13:i+1])
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe (wait for 1d close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 12h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Chop (14), Donchian (20), volume MA (20)
    start_idx = max(14, 19, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Regime filter based on Choppiness Index
        chop_val = chop_aligned[i]
        trending_regime = chop_val < 38.2   # Trending market
        ranging_regime = chop_val > 61.8    # Ranging market
        
        if position == 0:
            # In trending regime: follow Donchian breakout
            if trending_regime:
                # Long: break above Donchian high with volume
                if price > donchian_high[i] and vol_filter:
                    signals[i] = size
                    position = 1
                # Short: break below Donchian low with volume
                elif price < donchian_low[i] and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            # In ranging regime: mean revert at Donchian levels
            elif ranging_regime:
                # Long: near Donchian low (support)
                if price <= donchian_low[i] * 1.005 and vol_filter:  # Within 0.5% of low
                    signals[i] = size
                    position = 1
                # Short: near Donchian high (resistance)
                elif price >= donchian_high[i] * 0.995 and vol_filter:  # Within 0.5% of high
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Chop between 38.2-61.8: no clear regime, stay flat
                signals[i] = 0.0
        elif position == 1:
            # Exit long: in trending regime when price touches Donchian low
            # In ranging regime when price reaches middle or opposite extreme
            if trending_regime:
                if price <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            elif ranging_regime:
                # Exit when price reaches midpoint or opposite extreme
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                if price >= midpoint or price >= donchian_high[i] * 0.995:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Exit short: in trending regime when price touches Donchian high
            # In ranging regime when price reaches middle or opposite extreme
            if trending_regime:
                if price >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            elif ranging_regime:
                # Exit when price reaches midpoint or opposite extreme
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                if price <= midpoint or price <= donchian_low[i] * 1.005:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Choppiness_Donchian_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0