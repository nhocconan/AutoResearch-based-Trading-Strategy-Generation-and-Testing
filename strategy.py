#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w ATR volatility filter (ATR10 < ATR30) and volume confirmation (>1.5x 20-bar avg)
    # Enter long on breakout above Donchian high, short on breakout below Donchian low
    # Exit when price crosses Donchian midpoint
    # Volatility filter ensures breakouts occur during low volatility (pre-breakout compression)
    # Works in bull (breakouts with trend) and bear (only volatility-aligned breaks taken).
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
    # Using 1w HTF for ATR reduces noise vs 1d and improves regime detection.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_window = 20
    donchian_high_1d = pd.Series(high_1d).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_1d = (donchian_high_1d + donchian_low_1d) / 2.0
    
    # Align 1d Donchian levels to 1d timeframe (no-op but for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    # Get 1w data for ATR-based volatility filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for 1w
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan  # first value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(10) and ATR(30) for 1w
    atr_10_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1w = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Align 1w ATR values to 1d timeframe
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    atr_30_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_30_1w)
    
    # Volatility filter: ATR(10) < ATR(30) (low volatility regime)
    vol_filter = atr_10_1w_aligned < atr_30_1w_aligned
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(atr_10_1w_aligned[i]) or np.isnan(atr_30_1w_aligned[i]) or np.isnan(avg_volume[i])):
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

name = "1d_1w_donchian_atr_vol_filter_volume_v2"
timeframe = "1d"
leverage = 1.0