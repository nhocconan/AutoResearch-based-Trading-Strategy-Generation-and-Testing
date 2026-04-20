#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter combined with 1-week Donchian channel breakout
# Chop > 61.8 = ranging market (mean revert at Donchian bands), Chop < 38.2 = trending (follow Donchian breakout)
# Weekly Donchian(20) provides major support/resistance; volume > 2x 20-period average confirms breakout strength
# Designed for low-frequency, high-conviction trades to minimize fee drag in ranging/choppy markets
# Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-week data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channel on weekly timeframe
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = high_max_20
    donchian_low = low_min_20
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Load daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr14 / (atr14 * 14)) / log10(14)
    # Avoid division by zero and handle edge cases
    chop_raw = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop = np.where((atr14 > 0) & (sum_tr14 > 0), chop_raw, 50.0)  # Default to neutral if invalid
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate price and volume arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        price = close[i]
        
        # Regime determination
        is_ranging = chop_val > 61.8   # Choppy/ranging market
        is_trending = chop_val < 38.2   # Trending market
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        if position == 0:
            if is_ranging and has_volume:
                # In ranging market: mean reversion at Donchian bands
                long_signal = (price <= donchian_low_aligned[i])
                short_signal = (price >= donchian_high_aligned[i])
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
            elif is_trending and has_volume:
                # In trending market: follow Donchian breakout
                long_signal = (price > donchian_high_aligned[i])
                short_signal = (price < donchian_low_aligned[i])
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: opposite Donchian band or regime shift to extreme ranging
            if price >= donchian_high_aligned[i] or chop_val > 70.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite Donchian band or regime shift to extreme ranging
            if price <= donchian_low_aligned[i] or chop_val > 70.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ChopRegime_1wDonchian_MeanRevTrend"
timeframe = "12h"
leverage = 1.0