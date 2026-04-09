#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v1
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and CHOP > 61.8 (ranging market -> mean reversion fade of breakout fails).
# Short when price breaks below Donchian(20) low with volume > 1.5x average and CHOP > 61.8.
# Uses 1d HTF for trend filter: only take longs when price > 1d 50 EMA, shorts when price < 1d 50 EMA.
# Discrete sizing 0.0, ±0.25 to minimize fee churn. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v1"
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
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr = pd.Series(high - low).rolling(window=1, min_periods=1).sum()  # True range approximation for chop
    # True range: max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(chop[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Choppiness regime: CHOP > 61.8 indicates ranging market (fade breakouts)
        chop_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian low or volume dries up or chop breaks down (trend start)
            if close[i] <= lowest_low[i] or not volume_confirmed or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian high or volume dries up or chop breaks down (trend start)
            if close[i] >= highest_high[i] or not volume_confirmed or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long breakout: price breaks above Donchian high AND above 1d 50 EMA
                if close[i] > highest_high[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian low AND below 1d 50 EMA
                elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals