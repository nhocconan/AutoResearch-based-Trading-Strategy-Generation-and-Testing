#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP < 38.2 = trending). Enters long on upper band breakout in trending regime; short on lower band breakout. Exits on opposite band touch. Uses 1d HTF for alignment safety only. Target: 19-50 trades/year (75-200 total over 4 years). Volume filters weak breakouts, chop regime avoids whipsaws in sideways markets, Donchian provides clear structure. Works in bull/bear by following institutional volume-driven breakouts in trending regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 20 * 4h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        # Avoid division by zero and handle edge cases
        chop = np.full_like(sum_atr, np.nan, dtype=float)
        valid = (range_max_min != 0) & (~np.isnan(sum_atr)) & (~np.isnan(range_max_min))
        chop[valid] = 100 * np.log10(sum_atr[valid] / np.log10(window) / range_max_min[valid])
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Multi-timeframe: 1d alignment (for safety, not used in logic)
    df_1d = get_htf_data(prices, '1d')
    # Dummy array for alignment (not used in actual logic)
    dummy_1d = np.zeros(len(df_1d))
    dummy_1d_aligned = align_htf_to_ltf(prices, df_1d, dummy_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
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