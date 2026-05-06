#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h KAMA trend + 4h Williams %R mean reversion + volume confirmation
# Long when 12h KAMA is rising (bullish trend) AND 4h Williams %R < -80 (oversold) AND volume > 1.5 * avg_volume(20)
# Short when 12h KAMA is falling (bearish trend) AND 4h Williams %R > -20 (overbought) AND volume > 1.5 * avg_volume(20)
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h KAMA provides smooth trend filtering with lag compensation
# 4h Williams %R (14) captures mean reversion in extreme zones
# Volume confirmation filters weak reversals
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)

name = "4h_12hKAMA_4hWilliamsR_Volume_v1"
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
    
    # Get 12h data ONCE before loop for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h KAMA (ER=10, fast=2, slow=30)
    close_series_12h = pd.Series(close_12h)
    change = abs(close_series_12h.diff(10))
    volatility = close_series_12h.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama_12h = [np.nan] * len(close_12h)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc.iloc[i] * (close_12h[i] - kama_12h[i-1])
    kama_12h = np.array(kama_12h)
    
    # Get 4h data ONCE before loop for Williams %R
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Williams %R (14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Align 12h KAMA to 4h timeframe (wait for completed 12h bar)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Align 4h Williams %R to 4h timeframe (wait for completed 4h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h KAMA rising (bullish trend) AND 4h Williams %R < -80 (oversold) AND volume confirmation
            if (kama_aligned[i] > kama_aligned[i-1] and 
                williams_r_aligned[i] < -80 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h KAMA falling (bearish trend) AND 4h Williams %R > -20 (overbought) AND volume confirmation
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  williams_r_aligned[i] > -20 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals