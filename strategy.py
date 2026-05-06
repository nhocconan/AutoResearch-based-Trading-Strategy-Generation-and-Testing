#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian breakout with volume confirmation and volume expansion filter
# Long when price breaks above 1-day Donchian upper channel (20-period high) with volume > 1.5x 20-period average
# Short when price breaks below 1-day Donchian lower channel (20-period low) with volume > 1.5x 20-period average
# Uses daily Donchian channels for key support/resistance levels, volume for confirmation
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "12h_1dDonchian20_Volume_Expansion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Donchian Channel (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1d, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Volume expansion: current volume > previous bar volume (momentum confirmation)
    volume_expansion = volume > np.roll(volume, 1)
    volume_expansion[0] = False  # First bar has no previous
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume confirmation and expansion
            if close[i] > upper_donchian[i] and volume_filter[i] and volume_expansion[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian with volume confirmation and expansion
            elif close[i] < lower_donchian[i] and volume_filter[i] and volume_expansion[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals