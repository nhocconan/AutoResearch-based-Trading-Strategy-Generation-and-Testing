#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud + 1d Trend Filter
# Hypothesis: Ichimoku system (Tenkan/Kijun cross + cloud filter) on 6h combined with
# 1d EMA trend filter provides high-probability entries in both bull and bear markets.
# The cloud acts as dynamic support/resistance, while TK crosses signal momentum shifts.
# Works in bull via bullish TK crosses above cloud, in bear via bearish TK crosses below cloud.
# Target: 20-40 trades/year to minimize fee drag.
name = "6h_ichimoku_1d_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Cloud (Kumo) boundaries
    # Senkou spans are plotted 26 periods ahead, so we shift them back for current price comparison
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll, handle with nan
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku calculations are valid
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(daily_ema_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR bearish TK cross
            if close[i] < cloud_bottom[i] or (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR bullish TK cross
            if close[i] > cloud_top[i] or (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: bullish TK cross above cloud with bullish daily trend
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # Bullish TK cross
                close[i] > cloud_top[i] and                           # Price above cloud
                close[i] > daily_ema_6h[i]):                          # Bullish daily trend
                position = 1
                signals[i] = 0.25
            # Enter short: bearish TK cross below cloud with bearish daily trend
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # Bearish TK cross
                  close[i] < cloud_bottom[i] and                          # Price below cloud
                  close[i] < daily_ema_6h[i]):                            # Bearish daily trend
                position = -1
                signals[i] = -0.25
    
    return signals