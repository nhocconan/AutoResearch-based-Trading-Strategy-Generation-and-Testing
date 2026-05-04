#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter + 12h Donchian breakout with volume confirmation
# Choppiness Index > 61.8 = ranging market (mean revert at Donchian bands)
# Choppiness Index < 38.2 = trending market (breakout continuation)
# Long: CHOP > 61.8 + price < lower Donchian(20) + volume spike
# Short: CHOP > 61.8 + price > upper Donchian(20) + volume spike
# Trend filter: only trade breakouts when 12h CHOP < 38.2 and price breaks Donchian bands
# Works in bull markets (trend breakouts) and bear markets (mean reversion in ranges)
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Chop_DonchianBreakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Choppiness Index calculation (same timeframe as prices)
    df_6h = prices  # prices is already 6h
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    high_roll_max = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_roll_min = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_roll_sum = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).sum().values
    
    chop = 100 * np.log10(atr_roll_sum / (high_roll_max - low_roll_min)) / np.log10(14)
    chop = np.where((high_roll_max - low_roll_min) == 0, 50, chop)  # avoid division by zero
    
    # Get 12h data for Donchian channels (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian(20) from prior completed 12h bar
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_20_shifted = np.roll(donch_high_20, 1)
    donch_low_20_shifted = np.roll(donch_low_20, 1)
    donch_high_20_shifted[0] = np.nan
    donch_low_20_shifted[0] = np.nan
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20_shifted)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(chop[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion in ranging market: fade at Donchian bands
            if chop[i] > 61.8:  # ranging market
                if close[i] < donch_low_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = 0.25
                    position = 1
                elif close[i] > donch_high_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = -0.25
                    position = -1
            # Breakout continuation in trending market
            elif chop[i] < 38.2:  # trending market
                if close[i] > donch_high_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_low_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches opposite Donchian band OR Chop exits extreme range
            if close[i] > donch_high_aligned[i] or chop[i] < 38.2 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches opposite Donchian band OR Chop exits extreme range
            if close[i] < donch_low_aligned[i] or chop[i] < 38.2 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals