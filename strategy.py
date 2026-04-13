#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
    # Long: price breaks above 20-period high + volume > 1.5x 20-period avg + chop > 61.8 (range)
    # Short: price breaks below 20-period low + volume > 1.5x 20-period avg + chop > 61.8 (range)
    # Exit: price crosses 10-period midpoint (mean reversion in range)
    # Uses 1d data for volume and chop filters to avoid noise on 12h
    # Donchian breakouts capture momentum in ranging markets
    # Volume spike confirms institutional interest
    # Chop filter ensures we only trade in ranging regimes where mean reversion works
    # Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d volume spike filter (volume > 1.5x 20-period avg)
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma)
    
    # Calculate 1d chop regime filter (chop > 61.8 = ranging)
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (HHH - LLL)))
    atr_1 = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1[0] = high_1d[0] - low_1d[0]  # first bar
    sum_atr = pd.Series(atr_1).rolling(window=20, min_periods=20).sum().values
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    chop = 100 * np.log10(sum_atr) / (np.log10(20) * np.log10((highest_high_20 - lowest_low_20) + 1e-10))
    chop = np.where((highest_high_20 - lowest_low_20) == 0, 100, chop)  # avoid div by zero
    chop_regime = chop > 61.8  # ranging market
    
    # Align HTF filters to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close_12h[i] > highest_high[i]
        breakout_down = close_12h[i] < lowest_low[i]
        
        # HTF filters
        volume_confirmed = vol_spike_aligned[i] > 0.5
        chop_filter = chop_regime_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = breakout_up and volume_confirmed and chop_filter and position != 1
        short_entry = breakout_down and volume_confirmed and chop_filter and position != -1
        
        # Exit conditions: mean reversion to midpoint
        exit_long = position == 1 and close_12h[i] < donchian_mid[i]
        exit_short = position == -1 and close_12h[i] > donchian_mid[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_volume_chop_filter_v1"
timeframe = "12h"
leverage = 1.0