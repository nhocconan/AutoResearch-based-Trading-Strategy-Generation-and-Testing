#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian breakouts with daily ADX trend filter and volume confirmation.
# Long when price breaks above weekly Donchian upper with daily ADX > 20 and volume > 1.5x average.
# Short when price breaks below weekly Donchian lower with daily ADX > 20 and volume > 1.5x average.
# Exit when price returns to weekly Donchian midpoint or daily ADX drops below 15.
# Uses weekly Donchian for structure, daily ADX for trend strength, volume for confirmation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-week)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Load daily data ONCE for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, plus_dm14 / tr14 * 100, 0)
    minus_di14 = np.where(tr14 != 0, minus_dm14 / tr14 * 100, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Align indicators to lower timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 30)  # Need weekly Donchian and daily ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 20 indicates strong trend
        strong_trend = adx_aligned[i] > 20
        weak_trend = adx_aligned[i] < 15  # Exit when trend weakens
        
        if position == 0:
            # Look for Donchian breakouts with strong trend
            # Long: price breaks above weekly Donchian upper AND strong trend
            if (close[i] > donchian_upper_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly Donchian lower AND strong trend
            elif (close[i] < donchian_lower_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly Donchian midpoint or trend weakens
            if (close[i] <= donchian_mid_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly Donchian midpoint or trend weakens
            if (close[i] >= donchian_mid_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WeeklyDonchian_DailyADX_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0