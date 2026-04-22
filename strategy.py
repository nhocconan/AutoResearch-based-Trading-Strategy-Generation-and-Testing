#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + 1d Choppiness regime
# Long when price breaks above 20-period Donchian high + volume spike + CHOP > 61.8 (range) for mean reversion
# Short when price breaks below 20-period Donchian low + volume spike + CHOP > 61.8 (range) for mean reversion
# Exit when price returns to Donchian midpoint or CHOP < 38.2 (trending)
# Designed for low trade frequency (~20-50/year) with edge in range-bound markets
# Works in both bull (buy dips in range) and bear (sell rallies in range) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and Choppiness index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR for Choppiness index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 14-period highest high and lowest low for Choppiness denominator
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    # For efficiency, we calculate sum of ATR over 14 periods
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hl_range_14 = hh_14 - ll_14
    # Avoid division by zero
    hl_range_14 = np.where(hl_range_14 == 0, 1e-10, hl_range_14)
    chop = 100 * np.log10(atr_sum_14 / hl_range_14) / np.log10(14)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        chop_val = chop_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = volume > 1.5 * vol_ma_val
        
        # Range filter: Chop > 61.8 indicates ranging market (good for mean reversion)
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + volume spike + ranging market
            if price > donch_high_val and vol_spike and is_ranging:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + volume spike + ranging market
            elif price < donch_low_val and vol_spike and is_ranging:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian midpoint OR market starts trending (CHOP < 38.2)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to midpoint or market trends
                if price <= donch_mid_val or chop_val < 38.2:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to midpoint or market trends
                if price >= donch_mid_val or chop_val < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume_Spike_CHOP_Range"
timeframe = "4h"
leverage = 1.0