#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with 1w EMA trend filter and volume confirmation
# Long when price breaks above 1w Donchian upper (20) AND 1w EMA34 > EMA89 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian lower (20) AND 1w EMA34 < EMA89 AND volume > 1.5 * avg_volume(20)
# Exit when price touches 1w Donchian midpoint or opposite Donchian level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong structural breakout levels aligned with weekly session
# 1w EMA filter ensures alignment with weekly trend, reducing counter-trend trades in bear markets
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "1d_1wDonchian20_1wEMATrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:  # Need sufficient data for Donchian(20) and EMA89
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_upper_1w = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = low_series_1w.rolling(window=20, min_periods=20).min().values
    donchian_middle_1w = (donchian_upper_1w + donchian_lower_1w) / 2.0
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_1w)
    
    # Calculate 1w EMA34 and EMA89
    close_series_1w = pd.Series(close_1w)
    ema_34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = close_series_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA values to 1d timeframe (wait for completed 1w bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
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
            # Long: price breaks above 1w Donchian upper with 1w EMA34 > EMA89 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with 1w EMA34 < EMA89 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1w Donchian middle or lower (reversal or profit take)
            if close[i] <= donchian_middle_aligned[i] or close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1w Donchian middle or upper (reversal or profit take)
            if close[i] >= donchian_middle_aligned[i] or close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals