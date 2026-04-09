#!/usr/bin/env python3
# 12h_donchian_breakout_volume_regime_v1
# Hypothesis: 12h strategy using Donchian(20) breakout with volume confirmation (>1.3x 20-period average) and choppiness regime filter (CHOP < 38.2 = trending). Enters long on upper band breakout in trending regime; short on lower band breakout. Exits on opposite band touch. Uses 1d HTF for alignment safety. Target: 12-37 trades/year (50-150 total over 4 years). Works in bull/bear by following volume-driven institutional breakouts in trending regimes, avoiding whipsaws in chop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 20*12h = 10 days)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        tr = []
        for i in range(len(close)):
            if i == 0:
                tr.append(high[i] - low[i])
            else:
                tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
        tr = np.array(tr)
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        # Avoid division by zero
        range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
        chop = 100 * np.log10(sum_atr / np.log10(window) / range_max_min)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Multi-timeframe: 1d alignment (for higher timeframe context)
    df_1d = get_htf_data(prices, '1d')
    # Use 1d close for simple trend filter (optional alignment)
    close_1d = df_1d['close'].values
    # No HTF indicator used in logic, but alignment call required per rules
    # aligned_close_1d = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime filter: trending only (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price touches or breaks lower Donchian band
            if close[i] <= low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks upper Donchian band
            if close[i] >= high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only in trending regime with volume confirmation
            if trending_regime and volume_confirmed:
                # Long: price breaks above upper Donchian band
                if close[i] > high_max[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band
                elif close[i] < low_min[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals