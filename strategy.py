#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter.
# Long when price breaks above upper Donchian(20) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range).
# Short when price breaks below lower Donchian(20) AND 1d volume > 1.5x 20-period average AND chop > 61.8.
# Exit when price crosses the 12h midpoint (upper+lower)/2 OR chop < 38.2 (trend).
# Uses discrete position size 0.25. Designed to capture mean-reversion breakouts in ranging markets.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channels (20-period) ===
    # Upper and lower bands based on last 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint_donchian = (highest_20 + lowest_20) / 2
    
    # === 1d HTF: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === 1d HTF: Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh_ll_1d = hh_1d - ll_1d
    # Avoid division by zero
    hh_ll_1d_safe = np.where(hh_ll_1d == 0, 1e-10, hh_ll_1d)
    chop_1d = 100 * np.log10(sum_tr_14 / hh_ll_1d_safe) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d), 50, chop_1d)  # fill NaN with neutral
    
    chop_range = chop_1d > 61.8  # ranging market
    chop_trend = chop_1d < 38.2  # trending market (exit)
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 14 for CHOP)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(midpoint_donchian[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5
        in_range = chop_range_aligned[i] > 0.5
        in_trend = chop_trend_aligned[i] > 0.5
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint OR market trends (chop < 38.2)
            if price < midpoint_donchian[i] or in_trend:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint OR market trends (chop < 38.2)
            if price > midpoint_donchian[i] or in_trend:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_range:
            # LONG: Price breaks above upper Donchian AND volume spike AND ranging market
            if price > highest_20[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower Donchian AND volume spike AND ranging market
            elif price < lowest_20[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0