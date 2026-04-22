#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period) for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian upper and lower bands
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: price above upper = uptrend, below lower = downtrend
    trend_up = high_1w >= donchian_upper  # New weekly high
    trend_down = low_1w <= donchian_lower  # New weekly low
    
    # Align weekly trend to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down.astype(float))
    
    # Load daily data for entry signals (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = close_1d[0] - low_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily range for entry threshold
    daily_range = high_1d - low_1d
    
    # Align daily ATR and range to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(daily_range_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price breaks above weekly Donchian upper + volatility expansion
            if (trend_up_aligned[i] > 0.5 and 
                close[i] > donchian_upper[i-1] if i-1 >= 0 else False and  # Previous week's upper
                atr_14_aligned[i] > 1.5 * atr_14_aligned[max(0, i-1)] and  # Volatility expansion
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price breaks below weekly Donchian lower + volatility expansion
            elif (trend_down_aligned[i] > 0.5 and 
                  close[i] < donchian_lower[i-1] if i-1 >= 0 else False and  # Previous week's lower
                  atr_14_aligned[i] > 1.5 * atr_14_aligned[max(0, i-1)] and  # Volatility expansion
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Weekly trend reversal or volatility contraction
            if position == 1:
                # Exit long: Weekly downtrend signal or volatility contraction
                if (trend_down_aligned[i] > 0.5 or 
                    atr_14_aligned[i] < 0.8 * atr_14_aligned[max(0, i-1)]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Weekly uptrend signal or volatility contraction
                if (trend_up_aligned[i] > 0.5 or 
                    atr_14_aligned[i] < 0.8 * atr_14_aligned[max(0, i-1)]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyDonchianTrend_VolatilityExpansion"
timeframe = "6h"
leverage = 1.0