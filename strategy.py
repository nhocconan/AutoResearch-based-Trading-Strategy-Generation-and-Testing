#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Uses weekly Camarilla levels (R4/S4) from 1w timeframe for primary trend bias.
# Long when price breaks above 6h Donchian upper channel with volume spike and weekly R4 broken (bullish regime).
# Short when price breaks below 6h Donchian lower channel with volume spike and weekly S4 broken (bearish regime).
# Volume spike (>1.8x 24-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via weekly Camarilla R4/S4 regime filter.

name = "6h_Donchian20_1wCamarillaR4S4_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly Camarilla regime filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (R4/S4 are strong breakout/continuation levels)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = H - L
    range_1w = high_1w - low_1w
    # Camarilla levels: R4 = pivot + range * 1.1/2, S4 = pivot - range * 1.1/2
    R4_1w = pivot_1w + range_1w * 1.1 / 2.0
    S4_1w = pivot_1w - range_1w * 1.1 / 2.0
    
    # Align weekly Camarilla levels to 6h timeframe
    R4_1w_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    S4_1w_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Calculate 6h Donchian channels (20-bar)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume spike: >1.8x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for Donchian and weekly alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(R4_1w_aligned[i]) or 
            np.isnan(S4_1w_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: weekly Camarilla R4/S4 breakout
        bullish_regime = close[i] > R4_1w_aligned[i]
        bearish_regime = close[i] < S4_1w_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_upper[i] and volume_spike[i]
        short_breakout = close[i] < donchian_lower[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or regime reversal
        long_exit = close[i] < donchian_lower[i] or close[i] < S4_1w_aligned[i]
        short_exit = close[i] > donchian_upper[i] or close[i] > R4_1w_aligned[i]
        
        # Handle entries and exits
        if long_breakout and bullish_regime and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and bearish_regime and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals