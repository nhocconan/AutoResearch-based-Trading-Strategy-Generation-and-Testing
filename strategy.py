#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and chop regime filter
# Long when price breaks above 1d Camarilla R3 level AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range)
# Short when price breaks below 1d Camarilla S3 level AND volume > 1.5 * avg_volume(20) AND chop > 61.8 (range)
# Exit when price returns to 1d Camarilla midpoint (PP)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla R3/S3 represent strong breakout levels from 1d structure
# Volume filter ensures institutional participation, reducing false breakouts
# Chop filter (range regime) improves performance in sideways markets like 2025 BTC/ETH
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "12h_1dCamarilla_R3_S3_Breakout_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed 1d bars for Camarilla
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 4.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate Choppiness Index regime filter on 12h
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high) - min(low)))
    # Simplified: CHOP > 61.8 = ranging market (good for mean reversion/breakouts in range)
    # CHOP < 38.2 = trending market
    atr_period = 14
    chop_period = 14
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_period)
    chop_regime = chop > 61.8  # Range regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 level with volume confirmation AND in range regime
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                volume_confirm[i] and chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 level with volume confirmation AND in range regime
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  volume_confirm[i] and chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla PP
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla PP
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals