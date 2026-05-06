#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper(20) AND 1d EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 12h Donchian lower(20) AND 1d EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 12h Donchian middle (mean of upper/lower)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides strong intermediate structure with clear breakout levels
# 1d EMA34/EMA89 filter ensures alignment with longer-term trend (avoids counter-trend trades)
# Volume confirmation filters weak breakouts (requires 2x average volume)
# Works in bull (breakouts above upper in uptrend) and bear (breakdowns below lower in downtrend)

name = "4h_12hDonchian20_1dEMA34Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (Upper + Lower) / 2
    high_series_12h = pd.Series(high_12h)
    low_series_12h = pd.Series(low_12h)
    donchian_upper_20_12h = high_series_12h.rolling(window=20, min_periods=20).max().values
    donchian_lower_20_12h = low_series_12h.rolling(window=20, min_periods=20).min().values
    donchian_middle_20_12h = (donchian_upper_20_12h + donchian_lower_20_12h) / 2.0
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h Donchian channels to 4h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20_12h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_20_12h)
    
    # Align 1d EMA indicators to 4h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(150, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper with 1d EMA34 > EMA89 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower with 1d EMA34 < EMA89 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h Donchian middle (trend weakening)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h Donchian middle (trend weakening)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals