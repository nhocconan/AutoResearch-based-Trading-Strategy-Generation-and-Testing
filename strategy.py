#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above weekly Donchian high(20) AND 1d EMA50 > EMA200 AND volume > 1.5 * avg_volume(24)
# Short when price breaks below weekly Donchian low(20) AND 1d EMA50 < EMA200 AND volume > 1.5 * avg_volume(24)
# Exit when price touches weekly Donchian midpoint or opposite 10-period Donchian level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Donchian provides strong structural support/resistance from multi-week consolidation
# 1d EMA50/EMA200 filter ensures alignment with daily trend, reducing counter-trend trades in bear markets
# Volume confirmation (1.5x) filters weak breakouts while allowing sufficient frequency
# Works in bull (trend continuation breakouts above weekly high) and bear (trend continuation breakdowns below weekly low)

name = "6h_WeeklyDonchian20_1dEMA50Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate weekly Donchian(20) - highest high and lowest low of past 20 weekly bars
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_high_20 = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series_1w.rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align weekly Donchian levels to 6h timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    
    # Calculate 10-period Donchian for exit levels (more sensitive)
    donchian_high_10 = high_series_1w.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series_1w.rolling(window=10, min_periods=10).min().values
    donchian_high_10_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_10)
    donchian_low_10_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_10)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA values to 6h timeframe (wait for completed 1d bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 24-period average volume (4h equivalent)
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high(20) with 1d EMA50 > EMA200 and volume confirmation
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                ema_50_aligned[i] > ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low(20) with 1d EMA50 < EMA200 and volume confirmation
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches weekly Donchian midpoint or 10-period low (profit take or reversal)
            if close[i] <= donchian_mid_aligned[i] or close[i] <= donchian_low_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches weekly Donchian midpoint or 10-period high (profit take or reversal)
            if close[i] >= donchian_mid_aligned[i] or close[i] >= donchian_high_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals