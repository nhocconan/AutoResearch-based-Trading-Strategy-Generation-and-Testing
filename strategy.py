#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from below in 1d uptrend (price > EMA50) with volume spike.
# Short when Williams %R crosses below -20 from above in 1d downtrend (price < EMA50) with volume spike.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years.
# Williams %R identifies overbought/oversold conditions, 1d EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading with
# the 1d trend, avoiding counter-trend whipsaws. Designed for 12h timeframe to minimize fee drag.

name = "12h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    
    # Align Williams %R to 15m timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        williams_r_val = williams_r_aligned[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND 1d uptrend AND volume spike
            if i > 100 and williams_r_val > -80 and williams_r_aligned[i-1] <= -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND 1d downtrend AND volume spike
            elif i > 100 and williams_r_val < -20 and williams_r_aligned[i-1] >= -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 from above (reversal signal)
            if i > 100 and williams_r_val < -50 and williams_r_aligned[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 from below (reversal signal)
            if i > 100 and williams_r_val > -50 and williams_r_aligned[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals