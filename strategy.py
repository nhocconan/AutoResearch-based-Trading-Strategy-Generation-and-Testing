#!/usr/bin/env python3
"""
6h_Adaptive_Regime_Donchian_Volume
Hypothesis: On 6h timeframe, use 1d Chop Index to detect regime: CHOP > 61.8 = range (mean revert at Donchian(20) bounds), CHOP < 38.2 = trend (breakout continuation). Enter long on Donchian(20) breakout with volume > 1.5x average in trending regime, short on breakdown. In ranging regime, fade at Donchian(20) edges with volume confirmation. Uses 1d regime filter to avoid whipsaws, designed for low trade frequency (15-25/year) to minimize fee drag while adapting to BTC/ETH bull/bear/range markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chop Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Chop Index (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Chop Index = 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    chop = 100 * (np.log10(atr_14) - np.log10(hh_14 - ll_14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 6h Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), Chop (14), volume MA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        chop_val = chop_aligned[i]
        donch_high = period20_high[i]
        donch_low = period20_low[i]
        
        # Regime classification
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        # Neutral zone (38.2-61.8) = no new entries, hold existing
        
        if position == 0:
            if is_trending:
                # Trend regime: breakout continuation
                long_signal = close[i] > donch_high and volume_confirm[i]
                short_signal = close[i] < donch_low and volume_confirm[i]
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Range regime: mean reversion at edges
                long_signal = close[i] <= donch_low and volume_confirm[i]
                short_signal = close[i] >= donch_high and volume_confirm[i]
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral regime: no new entries
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # Exit trend long on breakdown or volume drop
                if close[i] < donch_low or not volume_confirm[i]:
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Exit range long at opposite edge or middle
                if close[i] >= donch_high or abs(close[i] - (donch_high + donch_low)/2) < (donch_high - donch_low)*0.1:
                    signals[i] = 0.0
                    position = 0
            else:
                # Neutral regime exit
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # Exit trend short on breakout or volume drop
                if close[i] > donch_high or not volume_confirm[i]:
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Exit range short at opposite edge or middle
                if close[i] <= donch_low or abs(close[i] - (donch_high + donch_low)/2) < (donch_high - donch_low)*0.1:
                    signals[i] = 0.0
                    position = 0
            else:
                # Neutral regime exit
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Adaptive_Regime_Donchian_Volume"
timeframe = "6h"
leverage = 1.0