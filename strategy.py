#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands (20,2) for mean reversion with volume spike confirmation.
# Enter long when price touches lower BB and volume > 2x 20-bar average, short when price touches upper BB with volume confirmation.
# Exit when price reverts to middle BB (20-period SMA) or opposite BB touch occurs.
# Weekly Bollinger Bands provide dynamic support/resistance on higher timeframe, reducing false signals.
# Volume spike confirms institutional interest at key levels. Works in ranging markets (mean reversion) and 
# during trending markets (pullbacks to weekly mean). Uses discrete position sizing (0.25) to control risk.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WeeklyBollinger_MeanReversion_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20,2)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Middle Band: 20-period SMA
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Standard Deviation: 20-period
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    
    # Upper Band: SMA + (2 * std)
    upper_bb = sma_20 + (2 * std_20)
    
    # Lower Band: SMA - (2 * std)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align weekly Bollinger Bands to daily timeframe
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Calculate daily volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2 * volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band conditions
        price_at_upper = close[i] >= upper_bb_aligned[i]
        price_at_lower = close[i] <= lower_bb_aligned[i]
        price_at_middle = abs(close[i] - sma_20_aligned[i]) < (0.001 * sma_20_aligned[i])  # Within 0.1% of middle band
        
        # Entry conditions
        long_entry = price_at_lower and volume_confirm[i]
        short_entry = price_at_upper and volume_confirm[i]
        
        # Exit conditions: price reverts to middle band or opposite BB touch
        long_exit = price_at_middle or price_at_upper
        short_exit = price_at_middle or price_at_lower
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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