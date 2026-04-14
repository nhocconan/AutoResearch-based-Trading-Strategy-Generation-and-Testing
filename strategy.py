# 12h_Custom_Pivot_Trend_Filter
# Hypothesis: 12h strategy using custom pivot points (resistance/support) from 1-day high/low/close
# combined with 1-week EMA trend filter and volume confirmation. Long when price breaks above
# R2 pivot in bullish trend (price > weekly EMA), short when price breaks below S2 in bearish trend.
# Uses mean reversion within trend context with strict entry conditions to limit trades.
# Target: 50-150 total trades over 4 years for optimal balance between signal quality and frequency.

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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate custom pivot points from 1d data (R2, S2 levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate 1-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 12h bar (12h = 0.5 * 1d)
        idx_1d = i // 2
        if idx_1d < 20:  # Need sufficient 1d data for pivots
            continue
            
        # Get 1w index for current 12h bar (12h = 1/14 * 1w approximately)
        idx_1w = i // 14
        if idx_1w < 21:  # Need sufficient 1w data for EMA
            continue
        
        # Previous values to avoid look-ahead
        r2_prev = r2[idx_1d-1]
        s2_prev = s2[idx_1d-1]
        ema_prev = ema_1w[idx_1w-1]
        
        if np.isnan(r2_prev) or np.isnan(s2_prev) or np.isnan(ema_prev):
            continue
        
        # Create arrays for alignment (constant values for the period)
        r2_arr = np.full(len(df_1d), r2_prev)
        s2_arr = np.full(len(df_1d), s2_prev)
        ema_arr = np.full(len(df_1w), ema_prev)
        
        # Align to 12h timeframe
        r2_12h = align_htf_to_ltf(prices, df_1d, r2_arr)[i]
        s2_12h = align_htf_to_ltf(prices, df_1d, s2_arr)[i]
        ema_12h = align_htf_to_ltf(prices, df_1w, ema_arr)[i]
        
        if position == 0:
            # Long: price breaks above R2 + price above weekly EMA (bullish trend) + volume confirmation
            if (close[i] > r2_12h and  # price breaks above R2 resistance
                close[i] > ema_12h and  # price above weekly EMA (bullish trend)
                volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 + price below weekly EMA (bearish trend) + volume confirmation
            elif (close[i] < s2_12h and  # price breaks below S2 support
                  close[i] < ema_12h and  # price below weekly EMA (bearish trend)
                  volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price crosses below pivot or weekly EMA
            pivot_arr = np.full(len(df_1d), pivot[idx_1d-1])
            pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_arr)[i]
            if close[i] < pivot_12h or close[i] < ema_12h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price crosses above pivot or weekly EMA
            pivot_arr = np.full(len(df_1d), pivot[idx_1d-1])
            pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_arr)[i]
            if close[i] > pivot_12h or close[i] > ema_12h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Custom_Pivot_Trend_Filter"
timeframe = "12h"
leverage = 1.0