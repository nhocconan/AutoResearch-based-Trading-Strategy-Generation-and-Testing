#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w pivot-based regime filter and volume confirmation.
# Long: Close breaks above Donchian upper AND weekly pivot shows bullish bias (price above weekly pivot) AND volume > 1.5x 20-period MA
# Short: Close breaks below Donchian lower AND weekly pivot shows bearish bias (price below weekly pivot) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or weekly pivot bias flips or volume drops.
# Weekly pivot provides structural bias from higher timeframe, reducing false breakouts in ranging markets.
# Works in bull via long signals when above pivot and bear via short signals when below pivot.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).

name = "6h_Donchian20_1wPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot regime filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_pivot = (df_1w_high + df_1w_low + df_1w_close) / 3.0
    # Resistance 1: R1 = 2*P - L
    weekly_r1 = 2 * weekly_pivot - df_1w_low
    # Support 1: S1 = 2*P - H
    weekly_s1 = 2 * weekly_pivot - df_1w_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian channels (20-period) on 6h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        pivot_val = weekly_pivot_aligned[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine weekly pivot bias
        is_bullish_bias = close_val > pivot_val  # Price above weekly pivot = bullish bias
        is_bearish_bias = close_val < pivot_val  # Price below weekly pivot = bearish bias
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND bullish bias AND volume spike
            if close_val > donchian_upper[i] and is_bullish_bias and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower AND bearish bias AND volume spike
            elif close_val < donchian_lower[i] and is_bearish_bias and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian lower OR bias turns bearish OR volume drops
            if close_val < donchian_lower[i] or not is_bullish_bias or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian upper OR bias turns bullish OR volume drops
            if close_val > donchian_upper[i] or not is_bearish_bias or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals