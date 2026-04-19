#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volatility regime filter and volume confirmation
# Uses daily ATR ratio to distinguish trending vs ranging markets:
# - In trending markets (ATR ratio > 1.2): trade Donchian breakouts with volume
# - In ranging markets (ATR ratio <= 1.2): fade Donchian touches at bands with volume
# Adaptive to both bull and bear markets via volatility regime detection.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "4h_Donchian20_VolatilityRegime_Volume"
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
    
    # Get 1d data for ATR calculation (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) on daily
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_10 / atr_30  # >1.2 = trending, <=1.2 = ranging
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need ATR(30) and Donchian(20) data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Donchian levels
        upper = highest_20[i]
        lower = lowest_20[i]
        
        if position == 0:
            # Determine market regime
            is_trending = atr_ratio_val > 1.2
            
            if is_trending:
                # Trending market: trade breakouts
                if price > upper and volume_confirmed:
                    signals[i] = 0.30
                    position = 1
                elif price < lower and volume_confirmed:
                    signals[i] = -0.30
                    position = -1
            else:
                # Ranging market: fade Donchian touches
                if price >= upper * 0.995 and volume_confirmed:  # Near upper band
                    signals[i] = -0.30
                    position = -1
                elif price <= lower * 1.005 and volume_confirmed:  # Near lower band
                    signals[i] = 0.30
                    position = 1
        
        elif position == 1:
            # Exit long: price touches lower Donchian band or loses volume
            if price <= lower or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price touches upper Donchian band or loses volume
            if price >= upper or not volume_confirmed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals