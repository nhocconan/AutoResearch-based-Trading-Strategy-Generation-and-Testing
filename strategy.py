#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout (20-period) with 1d EMA50 trend filter, volume confirmation (>1.8x 20-bar average volume), and choppiness regime filter (CHOP < 38.2 = trending market). Uses discrete position sizing (0.30) to minimize fee churn. Designed to capture strong trends in both bull and bear markets while avoiding false breakouts in choppy conditions. Target: 75-150 total trades over 4 years.

name = "4h_Donchian20_1dEMA50_VolumeChopRegime_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) for breakout signals
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period, shifted to avoid look-ahead)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate Choppiness Index (CHOP) on 14-period for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop = np.where(true_range_sum == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback period
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, above 1d EMA50, volume spike, trending regime
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i] and 
                chop[i] < 38.2):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian lower band, below 1d EMA50, volume spike, trending regime
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i] and 
                  chop[i] < 38.2):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band OR chop becomes too high (choppy market)
            if (close[i] < lowest_low[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band OR chop becomes too high (choppy market)
            if (close[i] > highest_high[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals