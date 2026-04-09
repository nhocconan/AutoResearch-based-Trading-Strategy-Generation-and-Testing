#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v1
# Hypothesis: 12h strategy using Donchian channel breakouts with volume confirmation and choppiness regime filter.
# Long when price breaks above 20-bar Donchian high with volume > 1.5x average and CHOP > 61.8 (ranging market).
# Short when price breaks below 20-bar Donchian low with volume > 1.5x average and CHOP > 61.8.
# Uses 1d HTF data for Donchian channels and volume average, called ONCE before loop.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v1"
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
    
    # 1d HTF data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    volume_d = df_1d['volume'].values
    
    # 20-period Donchian channels
    donchian_high = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # 20-period average volume for confirmation
    volume_ma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) on 1d - range: 0-100, >61.8 = ranging, <38.2 = trending
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Chop = 100 * log10(sum(tr) / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Align all 1d data to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma_aligned[i]
        # Chop filter: CHOP > 61.8 indicates ranging market (good for mean reversion/breakouts in range)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian low or volume dries up or chop drops (trending starts)
            if close[i] <= donchian_low_aligned[i] or not volume_confirmed or not chop_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian high or volume dries up or chop drops (trending starts)
            if close[i] >= donchian_high_aligned[i] or not volume_confirmed or not chop_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_filter:
                # Long breakout: price breaks above Donchian high with volume AND chop filter
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian low with volume AND chop filter
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals