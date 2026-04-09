#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 4h choppiness regime filter
# - Primary signal: Donchian(20) breakout on 4h close (long: close > upper band, short: close < lower band)
# - Volume confirmation: 1d volume > 1.5 * 20-period average volume (ensures participation)
# - Regime filter: 4h choppiness index > 61.8 (range market) for mean reversion, < 38.2 for trend continuation
# - Exit: Donchian(10) opposite band touch or choppiness regime shift
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, volume confirms validity, chop filter adapts to regime

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    volume_1d = df_1d['volume'].values
    
    # 1d volume regime: volume > 1.5 * 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lower_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # 4h choppiness index
    def calculate_choppiness(high, low, close, window=14):
        atr = np.maximum(high - low, 
                         np.maximum(np.abs(high - np.roll(close, 1)), 
                                    np.absolute(np.roll(low, 1) - close)))
        atr = pd.Series(atr).rolling(window=window, min_periods=1).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=1).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=1).min().values
        chop = np.where(
            (highest_high - lowest_low) == 0,
            50.0,
            100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(window)
        )
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    chop_range = chop > 61.8  # range market
    chop_trend = chop < 38.2  # trending market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(upper_10[i]) or np.isnan(lower_10[i]) or
            np.isnan(chop[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian(10) OR chop shifts to strong trend
            if close[i] <= lower_10[i] or (chop_trend[i] and not chop_range[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian(10) OR chop shifts to strong trend
            if close[i] >= upper_10[i] or (chop_trend[i] and not chop_range[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian(20) upper AND volume spike AND in range regime
            if close[i] > upper_20[i] and volume_spike_aligned[i] and chop_range[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian(20) lower AND volume spike AND in range regime
            elif close[i] < lower_20[i] and volume_spike_aligned[i] and chop_range[i]:
                position = -1
                signals[i] = -0.25
    
    return signals