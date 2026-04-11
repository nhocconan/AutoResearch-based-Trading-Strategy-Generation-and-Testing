#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_breakout_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate Donchian channels on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Align daily Donchian channels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.5x 20-period average on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long condition: price breaks above daily Donchian high with volume
        long_signal = volume_confirmed and (price_high > donchian_high_aligned[i])
        
        # Short condition: price breaks below daily Donchian low with volume
        short_signal = volume_confirmed and (price_low < donchian_low_aligned[i])
        
        # Exit when price returns to the opposite Donchian level
        exit_long = position == 1 and (price_low < donchian_low_aligned[i])
        exit_short = position == -1 and (price_high > donchian_high_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily Donchian breakout with volume confirmation on 6h timeframe.
# Uses 20-period Donchian channels on daily timeframe to establish structural
# support/resistance levels. Enters long when 6h price breaks above daily Donchian
# high with volume confirmation (>1.5x average volume), short when breaks below
# daily Donchian low with volume. Exits when price returns to the opposite
# Donchian level. Works in both bull and bear markets by capturing breakouts
# from established ranges. Volume confirmation ensures breakouts are supported
# by market participation, reducing false signals. Target: 50-150 total trades
# over 4 years (12-37/year) to minimize fee drag on 6h timeframe. Donchian
# breakouts capture momentum while volume filter ensures quality signals.
# This approach has shown promise on higher timeframes and adapts well to 6h.