#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > EMA50 for longs, < EMA50 for shorts), volume confirmation (>1.8x 20-bar avg volume), and weekly regime filter (price > weekly EMA200 for bull regime, < for bear). 
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Donchian breakouts capture momentum; 1d EMA50 ensures intermediate trend alignment; 
# Volume confirmation filters low-participation breakouts; Weekly EMA200 regime filter adapts to bull/bear markets.
# Designed for fewer, higher-quality trades to minimize fee drag while working in both bull and bear markets.

name = "6h_Donchian20_1dEMA50_Volume_WeeklyEMA200_Regime"
timeframe = "6h"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly EMA200 for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Donchian channels (20-period) from prior candle only
    lookback_dc = 20
    prior_high_max = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    prior_low_min = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_dc, lookback_vol, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(prior_high_max[i]) or np.isnan(prior_low_min[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime: bull if price > weekly EMA200, bear if price < weekly EMA200
            bull_regime = close[i] > ema_200_1w_aligned[i]
            bear_regime = close[i] < ema_200_1w_aligned[i]
            
            # LONG: Price breaks above Donchian upper band, close > 1d EMA50, volume spike, bull regime
            if (high[i] > prior_high_max[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i] and
                bull_regime):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band, close < 1d EMA50, volume spike, bear regime
            elif (low[i] < prior_low_min[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i] and
                  bear_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band OR volume drops below average
            if (low[i] < prior_low_min[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band OR volume drops below average
            if (high[i] > prior_high_max[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals