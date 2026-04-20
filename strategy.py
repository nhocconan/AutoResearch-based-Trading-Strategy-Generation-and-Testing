#!/usr/bin/env python3
# 1d_1w_Stochastic_Oscillator_Reversal
# Hypothesis: Stochastic oscillator on daily timeframe identifies overbought/oversold conditions.
# Weekly trend filter ensures we only trade with the higher timeframe trend.
# Mean reversion from extreme stochastic levels with volume confirmation.
# Works in bull/bear via weekly trend filter - only trade pullbacks in trend direction.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Stochastic_Oscillator_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for stochastic calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate daily Stochastic Oscillator (14,3,3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close_1d - lowest_low) / (highest_high - lowest_low)
    
    # %D = 3-period SMA of %K
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all daily and weekly levels to daily timeframe
    k_percent_aligned = align_htf_to_ltf(prices, df_1d, k_percent)
    d_percent_aligned = align_htf_to_ltf(prices, df_1d, d_percent)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and stochastic warmup
        # Get values
        k_val = k_percent_aligned[i]
        d_val = d_percent_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(k_val) or np.isnan(d_val) or np.isnan(ema34_1w_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Stochastic oversold (<20) and turning up with volume confirmation and above weekly EMA34
            if (k_val < 20 and d_val < 20 and  # Oversold condition
                k_val > d_val and  # %K crossing above %D (bullish crossover)
                vol_ratio_val > 1.5 and  # Volume confirmation
                close_1d[i] > ema34_1w_val):  # Only long in weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: Stochastic overbought (>80) and turning down with volume confirmation and below weekly EMA34
            elif (k_val > 80 and d_val > 80 and  # Overbought condition
                  k_val < d_val and  # %K crossing below %D (bearish crossover)
                  vol_ratio_val > 1.5 and  # Volume confirmation
                  close_1d[i] < ema34_1w_val):  # Only short in weekly downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Stochastic overbought or %K crosses below %D
            if k_val > 80 or k_val < d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Stochastic oversold or %K crosses above %D
            if k_val < 20 or k_val > d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals