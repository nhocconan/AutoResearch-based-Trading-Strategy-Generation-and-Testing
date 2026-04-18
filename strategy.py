#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_Volume_Regime_v1
Hypothesis: Use price channel breakouts (Donchian upper/lower) combined with volume spikes and regime filtering (Choppiness Index) to capture strong directional moves in both bull and bear markets. Donchian channels provide objective breakout levels, volume confirms institutional participation, and Choppiness Index filters out choppy regimes where breakouts fail. Designed for low trade frequency (~25-35/year) to minimize fee decay while maintaining edge in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (14-period)
    # High CHOP (>61.8) = range/chop (avoid breakouts)
    # Low CHOP (<38.2) = trending (favor breakouts)
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div/0
    chop[np.isnan(chop)] = 50
    chop_low = chop < 38.2  # trending regime
    
    # 1d EMA trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_low[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_spike = volume_spike[i]
        is_trending = chop_low[i]
        ema_1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume, in trending regime, above daily EMA
            if price > donch_high_val and vol_spike and is_trending and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume, in trending regime, below daily EMA
            elif price < donch_low_val and vol_spike and is_trending and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: break below Donchian low or loss of momentum (below daily EMA)
            if price < donch_low_val or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: break above Donchian high or loss of momentum (above daily EMA)
            if price > donch_high_val or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_PriceChannel_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0