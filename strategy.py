#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-bar average) and choppiness regime filter (CHOP < 38.2 = trending). Enters long on upper band breakout in trending regime; short on lower band breakout in trending regime. Uses 12h HTF trend filter (HMA21) to align with higher timeframe direction. Exits on opposite band touch. Target: 20-50 trades/year (75-200 total over 4 years). Donchian provides clear structure, volume filters weak breakouts, chop regime avoids whipsaws in sideways markets. Works in bull/bear by following institutional volume-driven breakouts in trending regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v1"
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
    
    # Volume average for confirmation (20-period = 20 bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = []
        tr = []
        for i in range(len(close)):
            if i == 0:
                tr.append(high[i] - low[i])
            else:
                tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
            if i < window:
                atr.append(np.nan)
            else:
                atr.append(np.mean(tr[i-window+1:i+1]))
        atr = np.array(atr)
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        chop = 100 * np.log10(sum_atr / np.log10(window) / range_max_min)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Multi-timeframe: 12h HTF trend (HMA21)
    df_12h = get_htf_data(prices, '12h')
    hma_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(chop[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: trending only (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        # HTF trend filter: price above/below 12h HMA21
        hma_trend_up = close[i] > hma_12h_aligned[i]
        hma_trend_down = close[i] < hma_12h_aligned[i]
        
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
            # Enter only in trending regime with volume confirmation and HTF trend alignment
            if trending_regime and volume_confirmed:
                # Long: price breaks above upper Donchian band AND HTF trend up
                if close[i] > high_max[i] and hma_trend_up:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band AND HTF trend down
                elif close[i] < low_min[i] and hma_trend_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals