#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R extreme reversal with 1d EMA34/EMA200 trend filter and volume confirmation
# Long when 12h Williams %R < -80 (oversold) AND 1d EMA34 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when 12h Williams %R > -20 (overbought) AND 1d EMA34 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when 12h Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R provides mean-reversion edge in ranging markets, trend filter ensures alignment with higher timeframe trend
# Volume confirmation filters weak signals, works in both bull and bear markets

name = "4h_12hWilliamsR_Extreme_1dEMA34Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams %R and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = -100 * (highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h)
    # Handle division by zero (when high == low)
    williams_r_12h = np.where((highest_high_12h - lowest_low_12h) == 0, -50, williams_r_12h)
    
    # Calculate 12h EMA34 for exit signal
    close_series_12h = pd.Series(close_12h)
    ema_34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h Williams %R and EMA to 4h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Align 1d EMA indicators to 4h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with 1d EMA34 > EMA200 and volume confirmation
            if (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and 
                ema_34_1d_aligned[i] > ema_200_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with 1d EMA34 < EMA200 and volume confirmation
            elif (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and 
                  ema_34_1d_aligned[i] < ema_200_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum shift)
            if williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum shift)
            if williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals