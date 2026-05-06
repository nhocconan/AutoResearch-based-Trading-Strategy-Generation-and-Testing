#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 6h Donchian breakout with 1w EMA trend filter and volume confirmation
# Long when price breaks above 6h Donchian upper (20) AND 1w EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 6h Donchian lower (20) AND 1w EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when price touches 6h Donchian midpoint or opposite Donchian level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# 6h Donchian provides strong structural breakout levels
# 1w EMA filter ensures alignment with long-term trend, reducing counter-trend trades
# Volume confirmation filters weak breakouts (2.0x average volume)
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "6h_6hDonchian20_1wEMATrend_Volume"
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
    
    # Get 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_series_6h = pd.Series(high_6h)
    low_series_6h = pd.Series(low_6h)
    donchian_upper_6h = high_series_6h.rolling(window=20, min_periods=20).max().values
    donchian_lower_6h = low_series_6h.rolling(window=20, min_periods=20).min().values
    donchian_middle_6h = (donchian_upper_6h + donchian_lower_6h) / 2.0
    
    # Align 6h Donchian levels to 6h timeframe (wait for completed 6h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_6h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_6h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle_6h)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 and EMA89
    close_series_1w = pd.Series(close_1w)
    ema_34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = close_series_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA values to 6h timeframe (wait for completed 1w bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
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
            # Long: price breaks above 6h Donchian upper with 1w EMA34 > EMA89 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian lower with 1w EMA34 < EMA89 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 6h Donchian middle or lower (reversal or profit take)
            if close[i] <= donchian_middle_aligned[i] or close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 6h Donchian middle or upper (reversal or profit take)
            if close[i] >= donchian_middle_aligned[i] or close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals