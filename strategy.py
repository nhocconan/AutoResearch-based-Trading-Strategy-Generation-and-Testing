#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume Spike + Choppiness Filter
# Hypothesis: Donchian(20) breakouts capture momentum, volume spike confirms institutional participation,
# and choppiness filter avoids false signals in sideways markets. Works in both bull and bear regimes.
# 4h timeframe balances responsiveness and noise. Target: 20-50 trades/year (80-200 over 4 years).
name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 2.0)
    
    # Choppiness Index (14-period) on daily timeframe
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(daily_high - daily_low)
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low
    tr[0] = daily_high[0] - daily_low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop[0:13] = np.nan  # First 13 values invalid
    
    # Align chop to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_4h[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness filter: only trade when trending (CHOP < 38.2)
        if chop_4h[i] > 38.2:
            # In choppy market, flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or volume spike fails
            if close[i] < low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or volume spike fails
            if close[i] > high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price closes above Donchian upper band
                if close[i] > high_max[i]:
                    position = 1
                    signals[i] = 0.30
                # Short breakdown: price closes below Donchian lower band
                elif close[i] < low_min[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals