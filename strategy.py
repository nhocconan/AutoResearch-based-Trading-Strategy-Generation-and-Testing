#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with 12h EMA21 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper (20) AND 12h EMA21 > EMA50 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Donchian lower (20) AND 12h EMA21 < EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 12h EMA21 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides strong trend-following structure with clear breakout levels
# 12h EMA21/EMA50 filter ensures alignment with intermediate trend
# Volume confirmation filters weak breakouts
# Works in bull (breakouts above upper channel in uptrend) and bear (breakdowns below lower channel in downtrend)

name = "4h_12hDonchian20_12hEMA21Trend_Volume"
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
    
    # Get 12h data ONCE before loop for Donchian channels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_series_12h = pd.Series(high_12h)
    low_series_12h = pd.Series(low_12h)
    donchian_upper_20 = high_series_12h.rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = low_series_12h.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA21 and EMA50 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_21_12h = close_series_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20)
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper with 12h EMA21 > EMA50 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_21_aligned[i] > ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower with 12h EMA21 < EMA50 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_21_aligned[i] < ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA21 (trend reversal)
            if close[i] < ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA21 (trend reversal)
            if close[i] > ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals