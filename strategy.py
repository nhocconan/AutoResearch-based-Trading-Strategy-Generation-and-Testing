#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1w EMA200 trend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold) AND 1w EMA200 > EMA200 previous (uptrend) AND volume > 1.5 * avg_volume(50) on 6h
# Short when 1d Williams %R > -20 (overbought) AND 1w EMA200 < EMA200 previous (downtrend) AND volume > 1.5 * avg_volume(50) on 6h
# Exit when Williams %R returns to -50 (mean reversion to midpoint)
# Uses discrete sizing 0.25 to limit drawdown and reduce fee churn
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies exhaustion points that often precede reversals in both bull and bear markets
# 1w EMA200 ensures we trade with the dominant long-term trend filter
# Volume confirmation validates the exhaustion move while limiting false signals
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies)

name = "6h_WilliamsR_Extreme_1wEMA200_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14))
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need at least 200 completed weekly bars for EMA200
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 50-period average volume on 6h
    avg_volume_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * avg_volume_50)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(avg_volume_50[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), 1w EMA200 > EMA200 previous (uptrend), volume spike, in session
            if (williams_r_aligned[i] < -80 and 
                ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), 1w EMA200 < EMA200 previous (downtrend), volume spike, in session
            elif (williams_r_aligned[i] > -20 and 
                  ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion)
            if williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion)
            if williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals