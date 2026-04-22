#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA40 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets (common in 2025+),
# mean reversion at extremes works well. Trend filter ensures we only take counter-trend
# trades when higher timeframe trend is weak (using EMA40 slope). Volume confirms interest.
# Works in both bull/bear markets: mean reversion in ranges, trend following in strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R and 1w EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Align Williams %R to 1d timeframe (same timeframe, no alignment needed for same TF)
    # But we'll keep the pattern for consistency
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load 1w data for EMA40 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA40 on weekly
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    # Align EMA40 to 1d timeframe
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Volume confirmation: 20-period average on 1d
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start from 14 to allow Williams %R calculation
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above weekly EMA (weak downtrend) + volume
            if williams_r_aligned[i] < -80 and close[i] > ema_40_1w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below weekly EMA (weak uptrend) + volume
            elif williams_r_aligned[i] > -20 and close[i] < ema_40_1w_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -30 for long, -70 to -50 for short)
            if position == 1:
                # Exit long: Williams %R rises above -50
                if williams_r_aligned[i] > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Williams %R falls below -50
                if williams_r_aligned[i] < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_MeanReversion_1wEMA40_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0