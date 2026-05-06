#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot R3/S3 breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1d Camarilla R3 AND volume > 1.5 * avg_volume(20) AND choppiness < 61.8 (trending regime)
# Short when price breaks below 1d Camarilla S3 AND volume > 1.5 * avg_volume(20) AND choppiness < 61.8 (trending regime)
# Exit when price returns to 1d Camarilla midpoint (PP) or choppiness > 61.8 (range regime)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla R3/S3 represent strong breakout levels from 1d structure
# Volume spike filters for institutional participation in breakouts
# Choppiness filter avoids whipsaws in ranging markets
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "4h_1dCamarilla_R3_S3_Breakout_Volume_Chop"
timeframe = "4h"
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
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for Camarilla
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate choppiness regime filter on 4h: CHOP(14) < 61.8 = trending regime
    # Chop = 100 * log10(sum(ATR(1),14) / (log10(14) * (max(high,14) - min(low,14))))
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First period
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1_14 / (np.log10(14) * (max_high_14 - min_low_14)))
    chop = np.where((max_high_14 - min_low_14) == 0, 100, chop)  # Avoid division by zero
    chop_regime = chop < 61.8  # Trending regime
    
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
            # Long: price breaks above 1d Camarilla R3 level with volume spike AND trending regime
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                volume_confirm[i] and chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 level with volume spike AND trending regime
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  volume_confirm[i] and chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla pivot point (PP) OR choppiness > 61.8 (range regime)
            if close[i] <= pp_aligned[i] or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla pivot point (PP) OR choppiness > 61.8 (range regime)
            if close[i] >= pp_aligned[i] or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals