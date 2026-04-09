#!/usr/bin/env python3
# 12h_donchian_breakout_1d_volume_chop_v1
# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above 20-bar high with above-average volume and chop > 61.8 (range).
# Short when price breaks below 20-bar low with above-average volume and chop > 61.8 (range).
# Uses 1d HTF for volume average and chop calculation to avoid lower timeframe noise.
# Works in bull/bear: chop filter ensures we only trade in ranging markets where breakouts are meaningful.
# Target: 12-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d chop index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d = np.concatenate([np.full(14, np.nan), atr_1d])  # align length
    true_range_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_1d - min_low_1d
    chop_1d = 100 * np.log10(true_range_sum / np.where(chop_denominator == 0, 1, chop_denominator)) / np.log10(14)
    chop_1d = np.where(chop_denominator == 0, np.nan, chop_1d)
    
    # Align 1d indicators to 12h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation on 12h timeframe
    volume = prices['volume'].values
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_12h[i]) or np.isnan(volume_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop drops below 38.2 (trending)
            if close[i] < donchian_low[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop drops below 38.2 (trending)
            if close[i] > donchian_high[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current 12h volume > 1.5x 12h average
            volume_confirmed = volume[i] > 1.5 * volume_ma_12h[i]
            chop_filter = chop_1d_aligned[i] > 61.8
            
            if volume_confirmed and chop_filter:
                # Check for breakout
                if close[i] > donchian_high[i]:
                    # Break above Donchian high → long
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_low[i]:
                    # Break below Donchian low → short
                    position = -1
                    signals[i] = -0.25
    
    return signals