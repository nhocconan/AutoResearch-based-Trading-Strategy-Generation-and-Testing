#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# In ranging markets (2025+), price tends to revert from channel extremes; in trending markets, breakouts capture momentum.
# Volume confirmation filters false breakouts. Chop filter avoids whipsaws in sideways markets.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 75-200 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # 1d HTF data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for ATR calculation
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index formula
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((highest_high_14 - lowest_low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 50) OR trending (chop < 50)
        # We'll use chop > 30 to avoid extremely choppy markets
        chop_regime = chop_aligned[i] > 30
        
        if position == 1:  # Long position
            # Exit: price moves below midpoint of channel or volume dries up
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < midpoint or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above midpoint of channel or volume dries up
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > midpoint or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above upper Donchian channel
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below lower Donchian channel
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals