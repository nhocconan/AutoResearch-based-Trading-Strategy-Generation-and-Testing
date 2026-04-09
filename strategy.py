#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v3
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high in trending market (CHOP < 38.2) with volume > 1.5x average.
# Short when price breaks below Donchian(20) low in trending market with volume confirmation.
# Uses 1d HTF for additional trend filter (price > 1d EMA50 for longs, < for shorts).
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v3"
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
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr = pd.Series(high - low).rolling(window=1, min_periods=1).sum()  # True range approximation
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = highest_high_chop - lowest_low_chop
    chop = np.where(chop_denom != 0, -100 * np.log10(atr_sum / chop_denom) / np.log10(chop_period), 50)
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i]) or np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime: trending when CHOP < 38.2
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR trend filter fails
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR trend filter fails
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and trending_regime:
                # Long entry: price breaks above Donchian high AND above 1d EMA50
                if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below 1d EMA50
                elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals