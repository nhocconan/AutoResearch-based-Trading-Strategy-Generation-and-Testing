#!/usr/bin/env python3
# 4h_donchian_volume_atr_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter.
# Works in bull/bear: Donchian captures breakouts, volume confirms institutional participation,
# ATR(50) > ATR(100) ensures trending market (avoids chop). Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_atr_v3"
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
    
    # 1d HTF data for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(50) and ATR(100) for trend filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr50_1d = pd.Series(tr1d).rolling(window=50, min_periods=50).mean().values
    atr100_1d = pd.Series(tr1d).rolling(window=100, min_periods=100).mean().values
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    atr100_1d_aligned = align_htf_to_ltf(prices, df_1d, atr100_1d)
    
    # 4h Donchian(20) channels
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 4h volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(atr50_1d_aligned[i]) or np.isnan(atr100_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ATR(50) > ATR(100) indicates trending market
        trending = atr50_1d_aligned[i] > atr100_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian channel OR loss of trend
            if close[i] < lower_channel[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian channel OR loss of trend
            if close[i] > upper_channel[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and trending market
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed and trending:
                # Long: price breaks above upper Donchian channel
                if close[i] > upper_channel[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian channel
                elif close[i] < lower_channel[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals