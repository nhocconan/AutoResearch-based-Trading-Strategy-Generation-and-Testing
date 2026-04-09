#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v2
# Hypothesis: 4h Donchian breakout with 1d volume confirmation and choppiness regime filter.
# Long: price breaks above 20-period Donchian high + 1d volume > 1.5x 20-day average + chop > 61.8 (range)
# Short: price breaks below 20-period Donchian low + 1d volume > 1.5x 20-day average + chop > 61.8 (range)
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v2"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: current volume > 1.5x 20-day average
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = align_htf_to_ltf(prices, df_1d, volume_1d > 1.5 * volume_ma_1d)
    
    # 1d choppiness index: CHOP > 61.8 indicates ranging market (good for mean reversion/breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * 14)) / log10(14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low) / 14) / np.log10(14)
    chop_confirmed = align_htf_to_ltf(prices, df_1d, chop > 61.8)
    
    # 4h Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(volume_confirmed_1d[i]) or np.isnan(chop_confirmed[i]) or
            np.isnan(period20_high[i]) or np.isnan(period20_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR loss of volume/chop confirmation
            if close[i] < period20_low[i] or not (volume_confirmed_1d[i] and chop_confirmed[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR loss of volume/chop confirmation
            if close[i] > period20_high[i] or not (volume_confirmed_1d[i] and chop_confirmed[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume and chop confirmation
            if volume_confirmed_1d[i] and chop_confirmed[i]:
                # Long: price breaks above Donchian high
                if close[i] > period20_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < period20_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals