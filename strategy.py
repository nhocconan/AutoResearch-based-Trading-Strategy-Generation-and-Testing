#!/usr/bin/env python3
# 1d_weekly_donchian_volume_regime_v1
# Hypothesis: 1d timeframe strategy using weekly Donchian breakouts with volume confirmation and choppiness regime filter.
# Long when price breaks above weekly Donchian high with volume > 1.5x average and chop < 61.8 (trending).
# Short when price breaks below weekly Donchian low with volume > 1.5x average and chop < 61.8.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 15-25 trades/year.
# Designed to work in both bull and bear markets via regime filter (avoid ranging markets) and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for Donchian channels and choppiness
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Weekly ATR for choppiness calculation (ATR14)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly true range sum for denominator (sum of TR over 14 periods)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Weekly choppiness index: CHOP = 100 * log10(TR_sum / (ATR * 14)) / log10(14)
    # Avoid division by zero and log of zero
    atr_times_14 = atr14_1w * 14
    chop_ratio = np.where((tr_sum > 0) & (atr_times_14 > 0), tr_sum / atr_times_14, np.nan)
    chop_1w = np.where((chop_ratio > 0) & ~np.isnan(chop_ratio), 100 * np.log10(chop_ratio) / np.log10(14), 50.0)
    chop_1w = np.nan_to_num(chop_1w, nan=50.0)
    
    # Align weekly choppiness to daily timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(chop_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low OR chop > 61.8 (ranging market)
            if close[i] < lowest_20_aligned[i] or chop_1w_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high OR chop > 61.8 (ranging market)
            if close[i] > highest_20_aligned[i] or chop_1w_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above weekly Donchian high with trending market (chop < 61.8)
                if close[i] > highest_20_aligned[i] and chop_1w_aligned[i] < 61.8:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below weekly Donchian low with trending market (chop < 61.8)
                elif close[i] < lowest_20_aligned[i] and chop_1w_aligned[i] < 61.8:
                    position = -1
                    signals[i] = -0.25
    
    return signals