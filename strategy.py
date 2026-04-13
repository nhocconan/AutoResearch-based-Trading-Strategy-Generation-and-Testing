#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and chop regime filter
    # Enter long when price breaks above 20-day high with volume > 1.5x 20-day avg and chop < 61.8
    # Enter short when price breaks below 20-day low with volume > 1.5x 20-day avg and chop < 61.8
    # Exit when price crosses opposite Donchian level or chop > 61.8 (range regime)
    # Uses 1w HTF for volume confirmation (more stable than 1d) and 1d for price action
    # Donchian breakouts capture strong trends; volume filter ensures participation
    # Chop filter avoids whipsaws in ranging markets (works in both bull and bear)
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for HTF volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Donchian high = highest high over past 20 days
    # Donchian low = lowest low over past 20 days
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR over 14 periods) / log10(highest high - lowest low over 14 periods))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index 0
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / np.log10(hh14 - ll14 + 1e-10)) / np.log10(10)
    # Handle division by zero and invalid values
    chop = np.where((hh14 - ll14) > 0, chop, 50.0)  # default to neutral when range is zero
    
    # Align 1w volume to 1d timeframe
    avg_volume_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    
    # Volume confirmation: current 1d volume > 1.5x 20-day average 1w volume
    volume_confirmed = volume > (1.5 * volume_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure Donchian channels are valid
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(chop[i]) or
            np.isnan(volume_1w_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]  # break above 20-day high
        breakout_down = close[i] < donchian_low[i]  # break below 20-day low
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop[i] < 61.8
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = breakout_up and volume_confirmed[i] and trending_regime and position != 1
        short_entry = breakout_down and volume_confirmed[i] and trending_regime and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_low[i] or chop[i] >= 61.8))
        exit_short = (position == -1 and (close[i] > donchian_high[i] or chop[i] >= 61.8))
        
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

name = "1d_1w_donchian_volume_chop_filter_v1"
timeframe = "1d"
leverage = 1.0