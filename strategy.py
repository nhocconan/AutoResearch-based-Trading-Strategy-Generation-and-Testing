#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and Choppiness Index regime filter.
# Long: Price breaks above Donchian upper band, volume > 1.5x 20-period average, CHOP(14) < 38.2 (trending).
# Short: Price breaks below Donchian lower band, volume > 1.5x 20-period average, CHOP(14) < 38.2 (trending).
# Exit: Opposite Donchian break or CHOP(14) > 61.8 (choppy regime).
# Uses 12h HTF for Donchian calculation to reduce noise, volume confirmation to filter weak breakouts,
# and Choppiness Index to avoid whipsaws in ranging markets. Designed to work in both bull and bear markets
# by only taking trades in trending regimes (CHOP < 38.2) and avoiding chop (CHOP > 61.8).
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for Donchian calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper and lower bands (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 4h timeframe (wait for completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate Choppiness Index on 4h (primary timeframe)
    # CHOP = 100 * log10(sum(ATR(1), n) / (max(high, n) - min(low, n))) / log10(n)
    # Where n = 14 period
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First TR is just high-low
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    
    sum_atr = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_market = chop[i] < 38.2
        
        # Exit choppy regime: CHOP > 61.8
        choppy_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR choppy regime
            if low[i] < donchian_low_aligned[i] or choppy_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR choppy regime
            if high[i] > donchian_high_aligned[i] or choppy_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, volume confirmed, trending market
            if (high[i] > donchian_high_aligned[i] and volume_confirmed and trending_market):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume confirmed, trending market
            elif (low[i] < donchian_low_aligned[i] and volume_confirmed and trending_market):
                position = -1
                signals[i] = -0.25
    
    return signals