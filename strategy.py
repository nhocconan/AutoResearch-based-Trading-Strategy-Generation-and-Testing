#!/usr/bin/env python3
# 12h_weekly_donchian_breakout_volume_v1
# Hypothesis: 12h strategy using weekly Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above weekly Donchian high (20) with volume > 1.8x 20-period average and chop < 61.8 (trending).
# Short when price breaks below weekly Donchian low (20) with volume > 1.8x 20-period average and chop < 61.8.
# Exit on opposite Donchian break or when chop > 61.8 (range) to avoid whipsaw.
# Uses weekly structure for direction, 12h for execution, volume and regime for filtration.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Donchian channels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Get daily data for choppiness regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR (14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes (14)
    abs_price_change = np.abs(np.diff(close_1d))
    abs_price_change = np.concatenate([[np.nan], abs_price_change])
    sum_abs_change = pd.Series(abs_price_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum(abs change) / (atr * 14)) / log10(14)
    chop = 100 * np.log10(sum_abs_change / (atr * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        # Regime filter: chop < 61.8 indicates trending market (good for breakouts)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: Price breaks below weekly Donchian low OR chop > 61.8 (range)
            if close[i] < donchian_low_aligned[i] or chop_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above weekly Donchian high OR chop > 61.8 (range)
            if close[i] > donchian_high_aligned[i] or chop_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Donchian break with volume confirmation and trending regime
            bullish_break = (close[i] > donchian_high_aligned[i]) and volume_confirmed and trending_regime
            bearish_break = (close[i] < donchian_low_aligned[i]) and volume_confirmed and trending_regime
            
            if bullish_break:
                position = 1
                signals[i] = 0.25
            elif bearish_break:
                position = -1
                signals[i] = -0.25
    
    return signals