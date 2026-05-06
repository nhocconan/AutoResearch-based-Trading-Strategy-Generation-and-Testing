#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1h Donchian breakout with 4h EMA trend filter and volume confirmation
# Long when price breaks above 1h Donchian upper (20) AND 4h EMA50 > EMA200 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1h Donchian lower (20) AND 4h EMA50 < EMA200 AND volume > 2.0 * avg_volume(20)
# Exit when price touches 1h Donchian midpoint or opposite Donchian level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1h Donchian provides intraday structural breakout levels for precise entries
# 4h EMA filter ensures alignment with intermediate trend, reducing counter-trend trades
# High volume threshold (2.0x) filters weak breakouts and reduces overtrading
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "4h_1hDonchian20_4hEMATrend_Volume"
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
    
    # Get 1h data ONCE before loop for Donchian channels
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h Donchian channels (20-period)
    high_series_1h = pd.Series(high_1h)
    low_series_1h = pd.Series(low_1h)
    donchian_upper_1h = high_series_1h.rolling(window=20, min_periods=20).max().values
    donchian_lower_1h = low_series_1h.rolling(window=20, min_periods=20).min().values
    donchian_middle_1h = (donchian_upper_1h + donchian_lower_1h) / 2.0
    
    # Align 1h Donchian levels to 4h timeframe (wait for completed 1h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1h, donchian_upper_1h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1h, donchian_lower_1h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1h, donchian_middle_1h)
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 and EMA200
    close_series_4h = pd.Series(close_4h)
    ema_50_4h = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = close_series_4h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA values to 4h timeframe (wait for completed 4h bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
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
            # Long: price breaks above 1h Donchian upper with 4h EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1h Donchian lower with 4h EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1h Donchian middle or lower (reversal or profit take)
            if close[i] <= donchian_middle_aligned[i] or close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1h Donchian middle or upper (reversal or profit take)
            if close[i] >= donchian_middle_aligned[i] or close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals