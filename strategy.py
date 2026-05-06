#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with 1d volume confirmation and ATR volatility filter
# Long when price breaks above 1w Donchian upper (20) AND volume > 1.3 * avg_volume(20) AND ATR(14) < ATR(50) (low vol breakout)
# Short when price breaks below 1w Donchian lower (20) AND volume > 1.3 * avg_volume(20) AND ATR(14) < ATR(50)
# Exit when price touches 1w Donchian midpoint
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong structural breakout levels aligned with weekly trend
# Volume confirmation filters weak breakouts
# ATR ratio filter ensures breakouts occur during low volatility periods (reducing false breakouts in choppy markets)
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "1d_1wDonchian20_Volume_ATRFilter"
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
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for Donchian(20)
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
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range calculation
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = tr.ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Ratio of short-term to long-term volatility
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with volume confirmation and low volatility (atr_ratio < 1.0)
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and atr_ratio[i] < 1.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with volume confirmation and low volatility (atr_ratio < 1.0)
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and atr_ratio[i] < 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1w Donchian middle (profit take or reversal)
            if close[i] <= donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1w Donchian middle (profit take or reversal)
            if close[i] >= donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals