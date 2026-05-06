#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R with 1w EMA trend filter and volume confirmation
# Long when 1d Williams %R crosses above -80 (oversold) AND 1w EMA34 > EMA89 AND volume > 1.5 * avg_volume(20)
# Short when 1d Williams %R crosses below -20 (overbought) AND 1w EMA34 < EMA89 AND volume > 1.5 * avg_volume(20)
# Exit when Williams %R crosses -50 (mean reversion) or opposite extreme
# Uses discrete sizing 0.25 to balance return and drawdown control
# Williams %R identifies reversal points in ranging markets, effective in both bull and bear regimes
# 1w EMA filter ensures alignment with weekly trend, reducing counter-trend trades
# Volume confirmation filters weak signals
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1dWilliamsR_1wEMATrend_Volume"
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
    if len(df_1d) < 14:  # Need sufficient data for Williams %R(14)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where(highest_high_1d == lowest_low_1d, -50, williams_r_1d)
    
    # Align 1d Williams %R to 4h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 and EMA89
    close_series_1w = pd.Series(close_1w)
    ema_34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = close_series_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA values to 4h timeframe (wait for completed 1w bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
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
            # Long: Williams %R crosses above -80 (oversold) with 1w EMA34 > EMA89 and volume confirmation
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) with 1w EMA34 < EMA89 and volume confirmation
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion) or crosses below -80 (stop)
            if williams_r_aligned[i] > -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion) or crosses above -20 (stop)
            if williams_r_aligned[i] < -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals