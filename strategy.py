#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation.
# Uses 1d Donchian(20) for directional breakout, 14-period Choppiness Index to filter ranging markets,
# and volume > 1.5x 20-period EMA for institutional participation. Designed to avoid whipsaws in chop
# while capturing strong trends in both bull and bear markets. Targets 20-40 trades/year.
name = "4h_ChopFilter_DonchianBreakout_1dVolume"
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
    
    # 1d data for Donchian channels and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for Donchian
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Choppiness Index (14-period)
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop formula: 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    # Align 1d indicators to 4h
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min_20)
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop_align[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Choppiness filter: only trade when trending (Chop < 38.2) or mean revert when ranging (Chop > 61.8)
        chop_val = chop_align[i]
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above Donchian high in trending market OR mean reversion from low in ranging market
            if (is_trending and price > donchian_high[i] and vol_spike[i]) or \
               (is_ranging and price < donchian_low[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in trending market OR mean reversion from high in ranging market
            elif (is_trending and price < donchian_low[i] and vol_spike[i]) or \
                 (is_ranging and price > donchian_high[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low (trending) or Donchian high (ranging)
            if is_trending and price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            elif is_ranging and price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high (trending) or Donchian low (ranging)
            if is_trending and price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            elif is_ranging and price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals