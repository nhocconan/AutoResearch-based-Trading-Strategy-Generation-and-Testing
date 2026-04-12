#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on daily data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate Chopiness Index on 1d data
    atr_period = 14
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - low_1d[:-1]),
            np.abs(low_1d[1:] - high_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Highest high and lowest low over atr_period
    hh_1d = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    ll_1d = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    chop_denom = hh_1d - ll_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(atr_1d * atr_period / chop_denom) / np.log10(atr_period)
    
    # Align Chop to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
        chop_value = chop_1d_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # Entry conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # In ranging markets: mean reversion at Donchian bounds
        # In trending markets: breakout continuation
        long_signal = False
        short_signal = False
        
        if is_ranging:
            # Mean reversion: buy at lower band, sell at upper band
            long_signal = close[i] <= donchian_low_aligned[i] and volume_ok[i]
            short_signal = close[i] >= donchian_high_aligned[i] and volume_ok[i]
        elif is_trending:
            # Trend following: breakout in direction of trend
            long_signal = long_breakout and volume_ok[i]
            short_signal = short_breakout and volume_ok[i]
        
        # Exit conditions
        exit_long = close[i] < donchian_low_aligned[i]  # Exit long if price breaks below lower band
        exit_short = close[i] > donchian_high_aligned[i]  # Exit short if price breaks above upper band
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals