#!/usr/bin/env python3
# 12h_donchian_20_volume_regime_v1
# Hypothesis: 12h Donchian(20) breakouts with volume confirmation and 1d chop regime filter work in both bull and bear markets by capturing breakouts while avoiding whipsaws in ranging conditions.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # 1d chop regime filter: Chop(14) > 61.8 = ranging (avoid), < 38.2 = trending (trade)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest and lowest over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high - lowest_low
    chop = 100 * np.log10(sum_tr_14 / hh_ll_diff) / np.log10(14)
    chop = np.where(hh_ll_diff == 0, 100, chop)  # avoid division by zero
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or chop becomes too high (ranging)
            if close[i] <= donchian_low[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or chop becomes too high (ranging)
            if close[i] >= donchian_high[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume confirmation and trending regime
            if close[i] > donchian_high[i] and volume[i] > vol_threshold[i] and chop_aligned[i] < 38.2:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation and trending regime
            elif close[i] < donchian_low[i] and volume[i] > vol_threshold[i] and chop_aligned[i] < 38.2:
                position = -1
                signals[i] = -0.25
    
    return signals