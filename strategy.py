#!/usr/bin/env python3
# mtf_1h_donchian_breakout_volume_4h1d_v1
# Hypothesis: 1h strategy using 4h Donchian(20) breakout with volume confirmation (>1.5x 20h average) and 1d choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend). Enters long on 4h upper band breakout in trending regime; short on lower band breakout in trending regime. Uses 1d HTF only for regime alignment safety. Target: 15-37 trades/year (60-150 total over 4 years). Donchian provides clear structure, volume filters weak breakouts, chop regime avoids whipsaws in sideways markets. Works in bull/bear by following institutional volume-driven breakouts in trending regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_breakout_volume_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h volume average for confirmation (20-period = 20 hours)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 4h Donchian channels (20-period = 20 * 4h = 80h ~ 3.3 days)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_max_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_max_4h_aligned = align_htf_to_ltf(prices, df_4h, high_max_4h)
    low_min_4h_aligned = align_htf_to_ltf(prices, df_4h, low_min_4h)
    
    # 1d Choppiness Index (14-period = 14 days)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(high_max_4h_aligned[i]) or np.isnan(low_min_4h_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20h average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: trending only (CHOP < 38.2)
        trending_regime = chop_1d_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price touches or breaks lower 4h Donchian band
            if close[i] <= low_min_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks upper 4h Donchian band
            if close[i] >= high_max_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter only in trending regime with volume confirmation and session
            if trending_regime and volume_confirmed:
                # Long: price breaks above upper 4h Donchian band
                if close[i] > high_max_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below lower 4h Donchian band
                elif close[i] < low_min_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals