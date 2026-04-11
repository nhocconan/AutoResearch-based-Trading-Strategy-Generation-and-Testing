#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly Donchian channels (20-week high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-period high and low for weekly
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed weekly bars
    donchian_high_20 = np.roll(donchian_high_20, 1)
    donchian_low_20 = np.roll(donchian_low_20, 1)
    donchian_high_20[0] = np.nan
    donchian_low_20[0] = np.nan
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after weekly Donchian warmup
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
        
        # Long conditions: Price breaks above weekly Donchian high with volume
        long_signal = volume_confirmed and (price_high > donchian_high_aligned[i])
        
        # Short conditions: Price breaks below weekly Donchian low with volume
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

# Hypothesis: Daily Donchian breakout from weekly channels with volume confirmation.
# Uses weekly Donchian channels (20-period high/low) as dynamic support/resistance.
# Enters long when daily price breaks above weekly Donchian high with volume confirmation
# (>1.5x average volume). Enters short when daily price breaks below weekly Donchian low
# with volume confirmation. Exits when price returns to the opposite Donchian level.
# Weekly timeframe provides structural context to avoid whipsaws, while daily
# timeframe captures breakouts. Volume confirmation ensures institutional participation.
# Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
# Works in both bull and bear markets by following the dominant trend on higher timeframe.