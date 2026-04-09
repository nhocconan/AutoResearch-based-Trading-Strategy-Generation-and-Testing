#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_chop_v4
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly chop regime filter.
# Long: Price breaks above 20-day high + volume > 2.0x 20-day average + weekly chop < 61.8 (trending)
# Short: Price breaks below 20-day low + volume > 2.0x 20-day average + weekly chop < 61.8 (trending)
# Exit: Price returns to opposite Donchian level (long exits at 20-day low, short exits at 20-day high)
# Uses 1d primary timeframe with 1w HTF for chop regime.
# Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_chop_v4"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-day)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for chop calculation
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - smoothed TR
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of absolute close changes (14-period)
    abs_close_chg = np.abs(np.diff(close_1w, prepend=np.nan))
    sum_abs_chg = pd.Series(abs_close_chg).rolling(window=14, min_periods=14).sum().values
    
    # Chopiness Index: 100 * log10(sum(atr14) / (atr14 * 14)) / log10(14)
    chop_raw = 100 * (np.log10(sum_abs_chg) - np.log10(atr_1w * 14)) / np.log10(14)
    chop = chop_raw  # Already in 0-100 range
    
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # Trending regime: weekly chop < 61.8
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to 20-day low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Price returns to 20-day high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above 20-day high + volume + trending regime
            if close[i] > donchian_high[i] and volume_confirmed and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below 20-day low + volume + trending regime
            elif close[i] < donchian_low[i] and volume_confirmed and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals