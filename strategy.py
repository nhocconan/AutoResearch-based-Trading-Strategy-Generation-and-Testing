#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme levels with 4h EMA34 trend filter and volume confirmation
# Long when price crosses above 1d Williams %R -80 (oversold) AND 4h EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when price crosses below 1d Williams %R -20 (overbought) AND 4h EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 4h EMA34 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R identifies overextended moves likely to reverse, providing edge in both bull and bear markets
# 4h EMA34/EMA89 filter ensures alignment with intermediate trend
# Volume confirmation filters weak breakouts
# Works in bull (buying oversold dips in uptrend) and bear (selling overbought rallies in downtrend)

name = "4h_1dWilliamsR_Extreme_4hEMA34Trend_Volume_v1"
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
    
    # Get 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Williams %R calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero when high == low
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 and EMA89 for trend filter
    close_series_4h = pd.Series(close_4h)
    ema_34_4h = close_series_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_4h = close_series_4h.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Align 4h EMA indicators to 4h timeframe (wait for completed 4h bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    ema_89_aligned = align_htf_to_ltf(prices, df_4h, ema_89_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above Williams %R -80 (oversold) with 4h EMA34 > EMA89 and volume confirmation
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Williams %R -20 (overbought) with 4h EMA34 < EMA89 and volume confirmation
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend reversal)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend reversal)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals