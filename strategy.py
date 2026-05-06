#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above weekly Donchian upper (20) AND 1d EMA34 > EMA89 AND volume > 1.3 * avg_volume(20)
# Short when price breaks below weekly Donchian lower (20) AND 1d EMA34 < EMA89 AND volume > 1.3 * avg_volume(20)
# Exit when price touches weekly Donchian midpoint
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Weekly Donchian provides strong structural breakout levels
# 1d EMA filter ensures alignment with medium-term trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "12h_weeklyDonchian20_1dEMATrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Upper = highest high over last 20 periods
    # Lower = lowest low over last 20 periods
    # Middle = (Upper + Lower) / 2
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_upper_1w = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = low_series_1w.rolling(window=20, min_periods=20).min().values
    donchian_middle_1w = (donchian_upper_1w + donchian_lower_1w) / 2.0
    
    # Align weekly Donchian levels to 12h timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_1w)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMA values to 12h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper with 1d EMA34 > EMA89 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower with 1d EMA34 < EMA89 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches weekly Donchian middle (profit take or reversal)
            if close[i] <= donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches weekly Donchian middle (profit take or reversal)
            if close[i] >= donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals