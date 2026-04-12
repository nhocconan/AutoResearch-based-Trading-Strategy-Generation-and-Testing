#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1w_donchian_breakout_volume_v1
# Weekly Donchian channel breakout with volume confirmation on 12h timeframe.
# Works in bull markets by catching breakouts above weekly highs, and in bear markets
# by catching breakdowns below weekly lows. Volume filter ensures breakouts have conviction.
# Low trade frequency expected (15-25/year) due to weekly channel width and volume requirement.
name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate volume moving average (20-period) and current volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if Donchian levels not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol_ratio[i] > 1.5
        
        # Breakout conditions
        bullish_breakout = (
            close[i] > donchian_high_aligned[i] and  # price breaks above weekly high
            volume_confirmed  # with volume confirmation
        )
        
        bearish_breakout = (
            close[i] < donchian_low_aligned[i] and  # price breaks below weekly low
            volume_confirmed  # with volume confirmation
        )
        
        # Exit conditions: opposite breakout or loss of momentum
        exit_long = (
            close[i] < donchian_low_aligned[i] and  # price breaks below weekly low
            volume_confirmed
        )
        
        exit_short = (
            close[i] > donchian_high_aligned[i] and  # price breaks above weekly high
            volume_confirmed
        )
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals