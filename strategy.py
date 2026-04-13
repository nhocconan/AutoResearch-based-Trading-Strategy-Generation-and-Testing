#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
    # Enter long on breakout above Donchian high, short on breakout below Donchian low
    # Exit when price crosses Donchian midpoint
    # Volatility filter ensures breakouts occur during low volatility (pre-breakout compression)
    # Works in bull (breakouts with trend) and bear (only volatility-aligned breaks taken).
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    # Using 1d HTF for ATR reduces noise vs 6h and improves regime detection.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_6h = pd.Series(high_6h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_6h = pd.Series(low_6h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_6h = (donchian_high_6h + donchian_low_6h) / 2.0
    
    # Align 6h Donchian levels to 6h timeframe (no-op but for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high_6h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low_6h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid_6h)
    
    # Get 1d data for ATR-based volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan  # first value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(10) and ATR(30) for 1d
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1d = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Align 1d ATR values to 6h timeframe
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    atr_30_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_30_1d)
    
    # Volatility filter: ATR(10) < ATR(30) (low volatility regime)
    vol_filter = atr_10_1d_aligned < atr_30_1d_aligned
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(atr_10_1d_aligned[i]) or np.isnan(atr_30_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with volatility filter and volume confirmation
        long_entry = breakout_up and vol_filter[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and vol_filter[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_mid_aligned[i])
        
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

name = "6h_1d_donchian_atr_vol_filter_volume_v1"
timeframe = "6h"
leverage = 1.0