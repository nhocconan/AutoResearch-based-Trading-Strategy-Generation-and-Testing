#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Donchian high(20) AND volume > 1.5x 20-period average AND CHOP(14) > 61.8 (ranging market).
# Short when price breaks below Donchian low(20) AND volume > 1.5x 20-period average AND CHOP(14) > 61.8.
# Exit when price crosses Donchian midpoint OR CHOP < 38.2 (trending regime).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts in ranging markets while avoiding false signals in strong trends.

name = "4h_DonchianBreakout_VolumeChop_1D_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if len(high) < lookback:
        return np.zeros(n)
    
    # Vectorized Donchian calculation
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d HTF data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = np.where(vol_ma_20 > 0, vol_1d / vol_ma_20, 0)  # ratio
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Choppiness Index (CHOP) - measures ranging vs trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) - smoothed TR
    atr_period = 14
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < atr_period:
            if i == 0:
                atr[i] = np.nan
            else:
                atr[i] = np.nanmean(tr[1:i+1]) if not np.all(np.isnan(tr[1:i+1])) else np.nan
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of ATR over lookback period
    chop_lookback = 14
    atr_sum = np.zeros_like(close_1d)
    for i in range(len(atr_sum)):
        if i < chop_lookback:
            atr_sum[i] = np.nan
        else:
            atr_sum[i] = np.nansum(atr[i-chop_lookback+1:i+1])
    
    # Max(high) - Min(low) over lookback period
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < chop_lookback:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.nanmax(high_1d[i-chop_lookback+1:i+1])
            min_low[i] = np.nanmin(low_1d[i-chop_lookback+1:i+1])
    
    # Chop = 100 * log10(sum(ATR) / (max_high - min_low)) / log10(lookback)
    range_val = max_high - min_low
    chop = np.zeros_like(close_1d)
    for i in range(len(chop)):
        if np.isnan(atr_sum[i]) or np.isnan(range_val[i]) or range_val[i] <= 0:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(atr_sum[i] / range_val[i]) / np.log10(chop_lookback)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND chop > 61.8 (ranging)
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] > 1.5 and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike AND chop > 61.8 (ranging)
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] > 1.5 and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses Donchian mid OR chop < 38.2 (trending regime)
            if (close[i] < donchian_mid[i] or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses Donchian mid OR chop < 38.2 (trending regime)
            if (close[i] > donchian_mid[i] or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals