#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long: Price breaks above 20-period Donchian high with volume > 1.5x 20-period average and chop > 61.8 (range).
# Short: Price breaks below 20-period Donchian low with volume > 1.5x 20-period average and chop > 61.8 (range).
# Exit: Price returns to opposite Donchian level or chop < 38.2 (trend) to avoid whipsaw.
# Uses 1d HTF for choppiness regime (CHOP > 61.8 = ranging market favorable for mean reversion breakouts).
# Target: 20-50 trades/year to minimize fee drag while capturing range breakouts in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR(14) for denominator
    tr_s = pd.Series(tr)
    atr_14 = tr_s.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods (numerator)
    tr_sum_14 = tr_s.rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR(14) / (ATR(14) * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    
    # Align Donchian, volume MA, and chop to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)  # Same timeframe
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma_aligned[i]
        # Choppiness regime: CHOP > 61.8 = ranging market (favorable for breakout mean reversion)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian low OR chop < 38.2 (trending market)
            if close[i] <= donchian_low_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian high OR chop < 38.2 (trending market)
            if close[i] >= donchian_high_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume and chop regime
            if (close[i] > donchian_high_aligned[i] and    # Break above Donchian high
                volume_confirmed and                       # Volume spike
                chop_regime):                              # Ranging market (chop > 61.8)
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume and chop regime
            elif (close[i] < donchian_low_aligned[i] and   # Break below Donchian low
                  volume_confirmed and                     # Volume spike
                  chop_regime):                            # Ranging market (chop > 61.8)
                position = -1
                signals[i] = -0.25
    
    return signals