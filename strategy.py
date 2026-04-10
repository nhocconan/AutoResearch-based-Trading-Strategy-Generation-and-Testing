#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Donchian(20) breakout captures medium-term momentum shifts on 12h chart
# - Long when price > Donchian(20) high AND volume > 2.0x 20-bar average AND chop > 61.8 (ranging)
# - Short when price < Donchian(20) low AND volume > 2.0x 20-bar average AND chop > 61.8 (ranging)
# - Exit when price reverts to Donchian(20) midline or opposite breakout occurs
# - Chop filter (61.8 threshold) ensures we trade mean reversion in ranging markets (2022-2025)
# - Volume confirmation filters false breakouts
# - Discrete position sizing (0.25) minimizes fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by trading ranges with volume confirmation

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 12h data
    period = 20
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate highest high and lowest low over the period
    donchian_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute Chopiness Index (14-period) on 1d data for regime filter
    # Chop > 61.8 = ranging market (good for mean reversion)
    # Chop < 38.2 = trending market
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high_1d[0] - low_1d[0]
    
    # Sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index = 100 * log10(sum(TR)/ (HH-LL)) / log10(14)
    chop = np.where(
        (hh_14 - ll_14) > 0,
        100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14),
        50  # Default when range is zero
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price > Donchian high with volume spike and chop > 61.8 (ranging)
            if (close_12h[i] > donchian_high[i] and 
                vol_spike.iloc[i] and 
                chop_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short signal: price < Donchian low with volume spike and chop > 61.8 (ranging)
            elif (close_12h[i] < donchian_low[i] and 
                  vol_spike.iloc[i] and 
                  chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midline
            # 2. Opposite breakout occurs
            if position == 1:  # Long position
                if close_12h[i] <= donchian_mid[i] or close_12h[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if close_12h[i] >= donchian_mid[i] or close_12h[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals