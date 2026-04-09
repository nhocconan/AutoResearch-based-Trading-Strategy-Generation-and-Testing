#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v3
# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and 1d chop regime filter.
# Long: price breaks above 20-period Donchian high + 1d volume > 1.5x 20-period average + 1d chop < 61.8
# Short: price breaks below 20-period Donchian low + 1d volume > 1.5x 20-period average + 1d chop < 61.8
# Exit: opposite Donchian breakout or chop > 61.8 (range regime)
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume confirmation: current volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d chop regime: CHOP(14) = 100 * log10(sum(ATR(1),14) / (log10(14) * (max(high,14)-min(low,14))))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_tr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = np.log10(14) * (max_high14 - min_low14)
    denominator = np.where(denominator == 0, 1e-10, denominator)  # avoid division by zero
    chop = 100 * np.log10(sum_tr14 / denominator)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 1d volume MA (aligned)
        volume_confirmed = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Chop regime: trending when CHOP < 61.8
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop > 61.8 (range regime)
            if close[i] < period20_low[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop > 61.8 (range regime)
            if close[i] > period20_high[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only in trending regime with volume confirmation
            if volume_confirmed and trending_regime:
                # Long: price breaks above Donchian high
                if close[i] > period20_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < period20_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals