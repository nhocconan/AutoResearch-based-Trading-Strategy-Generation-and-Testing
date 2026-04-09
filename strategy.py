#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v2
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Uses Donchian(20) breakouts for entry, volume > 1.5x 20-period average for confirmation,
# and Choppiness Index > 61.8 for ranging market (mean reversion) or < 38.2 for trending.
# In ranging markets (CHOP > 61.8): fade Donchian breakouts (short at upper band, long at lower band).
# In trending markets (CHOP < 38.2): follow Donchian breakouts (long at upper band, short at lower band).
# Weekly trend filter from 1d EMA200 to avoid counter-trend trades in bear markets.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v2"
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
    
    # 1d HTF data for weekly trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channels (20-period)
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr = pd.Series(high - low).rolling(window=1, min_periods=1).sum()  # True Range approximation
    # More accurate TR: max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    atr_mean = atr_sum / chop_period
    highest_max = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_min = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_max_min = highest_max - lowest_min
    range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
    chop = 100 * np.log10(atr_sum / (atr_mean * chop_period)) / np.log10(chop_period)
    chop = np.where(range_max_min == 0, 50, chop)  # neutral when no range
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(chop[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian lower band or volume dries up
            if close[i] <= lowest[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian upper band or volume dries up
            if close[i] >= highest[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Determine market regime
                is_ranging = chop[i] > 61.8
                is_trending = chop[i] < 38.2
                
                if is_ranging:
                    # In ranging market: fade Donchian breakouts
                    # Short at upper band, long at lower band
                    if close[i] >= highest[i] and close[i] < ema_200_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                    elif close[i] <= lowest[i] and close[i] > ema_200_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                elif is_trending:
                    # In trending market: follow Donchian breakouts
                    # Long at upper band, short at lower band
                    if close[i] >= highest[i] and close[i] > ema_200_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] <= lowest[i] and close[i] < ema_200_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals