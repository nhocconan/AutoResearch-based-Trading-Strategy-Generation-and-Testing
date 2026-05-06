#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Donchian breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper (20) AND 12h EMA50 > EMA200 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Donchian lower (20) AND 12h EMA50 < EMA200 AND volume > 1.5 * avg_volume(20)
# Exit when price touches 1d Donchian midpoint or opposite Donchian level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1d Donchian provides strong structural breakout levels
# 12h EMA filter ensures alignment with medium-term trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "6h_1dDonchian20_12hEMATrend_Volume"
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
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = highest high over last 20 periods
    # Lower = lowest low over last 20 periods
    # Middle = (Upper + Lower) / 2
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().values
    donchian_middle_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Align 1d Donchian levels to 6h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_1d)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = close_series_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA values to 6h timeframe (wait for completed 12h bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with 12h EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with 12h EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Donchian middle or lower (reversal or profit take)
            if close[i] <= donchian_middle_aligned[i] or close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1d Donchian middle or upper (reversal or profit take)
            if close[i] >= donchian_middle_aligned[i] or close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals