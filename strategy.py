#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and choppiness regime filter.
# Enter long when price breaks above Donchian(20) high, 1d volume > 2x 20-bar average, and choppiness < 38.2 (trending).
# Enter short when price breaks below Donchian(20) low, 1d volume > 2x 20-bar average, and choppiness < 38.2.
# Exit when price crosses Donchian midpoint.
# Uses discrete position sizing (0.25) to limit drawdown.
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fee drag.
# Donchian breakouts capture strong moves; volume confirms institutional participation; choppiness filter avoids whipsaws in ranging markets.
# Works in bull markets (catching breakouts) and bear markets (catching breakdowns) with volume confirmation reducing false signals.

name = "4h_Donchian20_VolumeSpike_ChopFilter_v1"
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
    
    # Get 1d data for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume and volume MA
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d).shift(1) - pd.Series(close_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    chop = np.where(hh_ll > 0, 100 * np.log10(sum_tr / hh_ll) / np.log10(14), 50)
    
    # Align 1d indicators to 4h
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: >2x 20-bar average volume
        vol_confirm = volume[i] > 2 * volume_ma_20_aligned[i]
        
        # Choppiness filter: < 38.2 indicates trending market
        chop_filter = chop_aligned[i] < 38.2
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Exit conditions: price crosses Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        # Handle entries and exits
        if breakout_up and vol_confirm and chop_filter and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and vol_confirm and chop_filter and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals