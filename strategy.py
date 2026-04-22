#!/usr/bin/env python3
"""
Hypothesis: Daily 200-day SMA trend filter combined with weekly Donchian channel breakouts.
Long when price > SMA200 and breaks above weekly Donchian upper (20-period).
Short when price < SMA200 and breaks below weekly Donchian lower.
Exit when price crosses back through the Donchian midpoint or violates SMA200.
Designed for low trade frequency by requiring both trend alignment and breakout.
Works in bull markets via long breakouts and bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily 200-day SMA for trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Load weekly data for Donchian channels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period Donchian channels on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align weekly Donchian levels to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(sma200[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > SMA200 and breaks above weekly Donchian upper
            if close[i] > sma200[i] and close[i] > donch_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < SMA200 and breaks below weekly Donchian lower
            elif close[i] < sma200[i] and close[i] < donch_low_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Donchian midpoint or violates SMA200 trend
            exit_signal = False
            
            if position == 1:
                # Exit long: price < Donchian midpoint or price < SMA200
                if close[i] < donch_mid_aligned[i] or close[i] < sma200[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price > Donchian midpoint or price > SMA200
                if close[i] > donch_mid_aligned[i] or close[i] > sma200[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_SMA200_WeeklyDonchian_Breakout"
timeframe = "1d"
leverage = 1.0