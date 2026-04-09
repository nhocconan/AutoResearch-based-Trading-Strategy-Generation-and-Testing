#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1_revised
# Hypothesis: 4h Donchian channel breakout with volume confirmation (>1.3x 20-period average) and 1d HTF choppiness regime filter (CHOP > 61.8 = range → mean revert at bands, CHOP < 38.2 = trend → follow breakout). Enters long on upper band breakout with volume confirmation in trending regime (CHOP < 38.2) or mean reversion from lower band in ranging regime (CHOP > 61.8). Short on lower band breakout with volume confirmation in trending regime or mean reversion from upper band in ranging regime. Uses discrete position sizing (0.25) to limit fee drag. Designed for moderate turnover (target: 20-50 trades/year) to work in both bull and bear markets by adapting to regime conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1_revised"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True range for 1d
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr1d = tr.rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum(atr1d,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1d / (max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks lower Donchian band (stoploss)
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks upper Donchian band (stoploss)
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            chop_val = chop_aligned[i]
            
            if volume_confirmed:
                # Trending regime (CHOP < 38.2): follow breakout
                if chop_val < 38.2:
                    # Long: price breaks above upper Donchian band
                    if close[i] > highest_high[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price breaks below lower Donchian band
                    elif close[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -0.25
                # Ranging regime (CHOP > 61.8): mean reversion at bands
                elif chop_val > 61.8:
                    # Long: price touches lower band and bounces up
                    if close[i] <= lowest_low[i] and close[i] > low[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price touches upper band and bounces down
                    elif close[i] >= highest_high[i] and close[i] < high[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals