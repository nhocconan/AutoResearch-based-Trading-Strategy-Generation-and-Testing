#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper band: highest high of last 20 days
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars
    upper_20 = np.roll(upper_20, 1)
    lower_20 = np.roll(lower_20, 1)
    upper_20[0] = np.nan
    lower_20[0] = np.nan
    
    # Align daily Donchian to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume filter: volume > 1.3x 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(40, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long when price breaks above daily upper band with volume
        long_signal = volume_confirmed and (price_high > upper_20_aligned[i])
        
        # Short when price breaks below daily lower band with volume
        short_signal = volume_confirmed and (price_low < lower_20_aligned[i])
        
        # Exit when price returns to the opposite band (mean reversion)
        exit_long = position == 1 and (price_low < lower_20_aligned[i])
        exit_short = position == -1 and (price_high > upper_20_aligned[i])
        
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

# Hypothesis: Daily Donchian breakout with volume confirmation on 4h timeframe.
# Uses 20-day Donchian channels (highest high/lowest low) from completed daily bars.
# Enters long when 4h price breaks above the daily upper band with volume >1.3x average.
# Enters short when 4h price breaks below the daily lower band with volume confirmation.
# Exits when price returns to the opposite band, capturing mean reversion after breakouts.
# Works in both bull and bear markets by trading breakouts in the direction of the trend.
# Volume confirmation ensures breakouts are supported by participation, reducing false signals.
# Target: 20-40 trades per year to minimize fee drag on 4h timeframe.