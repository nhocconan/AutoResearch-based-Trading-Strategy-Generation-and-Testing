#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + chop regime filter
# Donchian breakouts capture strong momentum moves. Volume confirms institutional participation.
# Chop regime filter (CHOP > 61.8) avoids false breakouts in ranging markets.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.
# Target: 20-50 trades/year on 4h to minimize fee drag.

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Chop Index (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop Index formula: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 20)  # Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        chop_regime = chop_aligned[i] > 61.8  # chop > 61.8 = ranging market (avoid breakouts)
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high, volume spike, NOT chop regime
            if price > donch_high[i] and vol_confirm and not chop_regime:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low, volume spike, NOT chop regime
            elif price < donch_low[i] and vol_confirm and not chop_regime:
                signals[i] = -0.30
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on Donchian low break or chop regime
            if price < donch_low[i] or chop_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit on Donchian high break or chop regime
            if price > donch_high[i] or chop_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals