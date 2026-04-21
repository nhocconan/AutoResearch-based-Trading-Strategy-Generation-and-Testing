#!/usr/bin/env python3
"""
4h_12h_1d_ThreeTimeframe_TripleFilter_v1
Hypothesis: Combine 4h Donchian breakout with 12h trend filter and 1d volume confirmation.
Long when price breaks above 4h Donchian upper channel (20) with 12h uptrend (close > EMA34) and 1d volume > 1.5x 20-day average.
Short when price breaks below 4h Donchian lower channel with 12h downtrend and volume confirmation.
Exit on opposite Donchian band touch. Uses three timeframes for confluence to reduce false signals.
Target: 20-40 trades/year per symbol. Designed to work in bull/bear by following 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        volume_ok = volume > 1.5 * vol_ma_1d_aligned[i]
        
        # 12h trend filter
        uptrend_12h = price > ema34_12h_aligned[i]
        downtrend_12h = price < ema34_12h_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high + 12h uptrend + volume
            if price > donch_high_20_aligned[i] and uptrend_12h and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + 12h downtrend + volume
            elif price < donch_low_20_aligned[i] and downtrend_12h and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: touch or cross below Donchian low
            if price < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: touch or cross above Donchian high
            if price > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1d_ThreeTimeframe_TripleFilter_v1"
timeframe = "4h"
leverage = 1.0