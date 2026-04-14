#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Confirmation and Chop Filter
# Uses Donchian(20) breakout for trend capture - proven to work in both bull/bear markets
# 1d Volume spike (volume > 1.5x 20-period average) confirms breakout strength
# 1d Choppiness Index filter: only trade when CHOP > 61.8 (ranging) for mean reversion or
# CHOP < 38.2 (trending) for trend continuation - avoids whipsaws in chop
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR) / (HHV - LLV)) / log10(period)
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = hh_14 - ll_14
    chop_raw = 100 * np.log10(sum_atr_14 / hh_ll_diff) / np.log10(14)
    chop = np.where(hh_ll_diff > 0, chop_raw, 50)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: current 4h volume > 1.5x 1d average volume
        vol_spike = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Chop regime: CHOP < 38.2 (trending) or CHOP > 61.8 (ranging)
        chop_val = chop_aligned[i]
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation
            if price > highest_20[i] and vol_spike and (is_trending or is_ranging):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian with volume confirmation
            elif price < lowest_20[i] and vol_spike and (is_trending or is_ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of Donchian channel or opposite breakout
            middle = (highest_20[i] + lowest_20[i]) / 2
            if price < middle or price < lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of Donchian channel or opposite breakout
            middle = (highest_20[i] + lowest_20[i]) / 2
            if price > middle or price > highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dVolume_ChopFilter"
timeframe = "4h"
leverage = 1.0