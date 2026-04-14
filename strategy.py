#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Choppiness Index as regime filter and 4h Donchian breakout for entry.
# Long when price breaks above Donchian(20) high AND Chop > 61.8 (ranging) for mean reversion bounce.
# Short when price breaks below Donchian(20) low AND Chop > 61.8 (ranging) for mean reversion fade.
# Exit when price crosses Donchian middle or Chop < 38.2 (trending regime) to avoid false signals.
# Chop > 61.8 indicates ranging market where mean reversion works; Chop < 38.2 indicates trending where breakouts work.
# This strategy exploits mean reversion in ranges and avoids trending markets that cause false breakouts.
# Works in both bull and bear markets by adapting to regime: mean revert in ranges, avoid trends.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need enough for Chop(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) = sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(ATR(14) / (HH(14) - LL(14))) / log10(14)
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # Small value to prevent div by zero
    chop = 100 * np.log10(atr_14 / hl_range) / np.log10(14)
    
    # Align Chop to 4h timeframe (with 2-bar delay for Chop confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    # Calculate Donchian Channels (20) on 4h data
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high_4h + lowest_low_4h) / 2
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need Chop(14) with 2-bar delay and Donchian(20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high_4h[i]) or
            np.isnan(lowest_low_4h[i]) or
            np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop_aligned[i] > 61.8
        
        # Trending regime filter: Chop < 38.2 indicates trending (avoid false signals)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 0:
            # Look for Donchian breakouts in ranging markets for mean reversion
            # Long: price breaks above Donchian high AND ranging market (expect pullback to middle)
            if (close[i] > highest_high_4h[i] and 
                ranging_market):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND ranging market (expect bounce to middle)
            elif (close[i] < lowest_low_4h[i] and 
                  ranging_market):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian middle OR market becomes trending
            if (close[i] < donchian_middle[i] or 
                trending_market):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses Donchian middle OR market becomes trending
            if (close[i] > donchian_middle[i] or 
                trending_market):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Choppiness_Donchian_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0