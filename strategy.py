#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + chop < 61.8
    # Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + chop < 61.8
    # Exit: price crosses Donchian midpoint or opposite breakout
    # Uses Donchian channels for structure, volume for conviction, chop filter to avoid whipsaws in ranging markets
    # Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend) with regime filter reducing false signals
    # Target: 50-150 total trades over 4 years (12-37/year) for optimal fee efficiency
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume and chop filters (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid_12h = (highest_high_12h + lowest_low_12h) / 2.0
    
    # Calculate 1d volume average (20-period)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d choppiness index (14-period)
    # Chop = 100 * log10(sum(ATR) / (log10(highest_high - lowest_low) * period)) / log10(period)
    tr1 = pd.Series(high_1d).rolling(window=2).max().values - pd.Series(low_1d).rolling(window=2).shift(1).values
    tr2 = abs(pd.Series(high_1d).rolling(window=2).shift(1).values - pd.Series(low_1d).rolling(window=2).shift(1).values)
    tr = np.maximum(tr1, tr2)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(highest_high_14 - lowest_low_14) * 14
    chop = np.where(
        (chop_denominator > 0) & (sum_atr_14 > 0),
        100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Align 1d indicators to 12h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_12h[i]
        breakout_down = close[i] < lowest_low_12h[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # We need to get the current 1d volume aligned to 12h
        # Since we don't have direct 1d volume in 12h alignment, we use the volume from prices
        # but check if it's elevated relative to the 1d volume average
        volume_confirm = volume[i] > 1.5 * volume_ma_1d_aligned[i]
        
        # Chop regime filter: avoid ranging markets (chop > 61.8 = ranging)
        chop_filter = chop_aligned[i] < 61.8
        
        # Entry conditions
        long_entry = breakout_up and volume_confirm and chop_filter and position != 1
        short_entry = breakout_down and volume_confirm and chop_filter and position != -1
        
        # Exit conditions: price crosses Donchian midpoint or opposite breakout
        exit_long = position == 1 and (close[i] < donchian_mid_12h[i] or breakout_down)
        exit_short = position == -1 and (close[i] > donchian_mid_12h[i] or breakout_up)
        
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