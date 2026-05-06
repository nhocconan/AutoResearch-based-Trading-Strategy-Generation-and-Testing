#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extreme levels with volume confirmation
# Long when price closes above 1w Williams %R oversold (-80) AND volume > 1.5 * avg_volume(20)
# Short when price closes below 1w Williams %R overbought (-20) AND volume > 1.5 * avg_volume(20)
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme touched
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Williams %R identifies overextended moves; volume confirms institutional participation
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets
# 1d timeframe minimizes fee drag while capturing multi-day trends

name = "1d_1wWilliamsR_Extreme_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 completed 1w bars for Williams %R
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HHV - Close) / (HHV - LLV) where period=14
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    # Handle division by zero when HHV == LLV
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1w Williams %R to 1d timeframe (wait for completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) with volume confirmation
            if (wr > -80 and williams_r_aligned[i-1] <= -80 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) with volume confirmation
            elif (wr < -20 and williams_r_aligned[i-1] >= -20 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 or reaches -20 (overbought)
            if wr >= -50 or wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 or reaches -80 (oversold)
            if wr <= -50 or wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals