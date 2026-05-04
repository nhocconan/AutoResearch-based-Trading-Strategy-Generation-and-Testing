#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Uses Ichimoku components (Tenkan, Kijun, Senkou Span A/B) from 6h for entry signals,
# 1d EMA50 for trend filter to avoid counter-trend trades, and volume spike for confirmation.
# Designed for 12-30 trades/year on 6h timeframe to minimize fee drag.
# Works in bull markets via bullish TK crosses above cloud and in bear markets via bearish TK crosses below cloud.
# The 1d EMA50 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "6h_Ichimoku_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Shift Senkou Spans forward by 26 periods
    senkou_a_leading = np.roll(senkou_a, 26)
    senkou_b_leading = np.roll(senkou_b, 26)
    senkou_a_leading[:26] = np.nan
    senkou_b_leading[:26] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or
            np.isnan(kijun[i]) or
            np.isnan(senkou_a_leading[i]) or
            np.isnan(senkou_b_leading[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_leading[i], senkou_b_leading[i])
        cloud_bottom = min(senkou_a_leading[i], senkou_b_leading[i])
        
        if position == 0:
            # Long conditions: bullish TK cross (Tenkan > Kijun) AND price above cloud AND above 1d EMA50 AND volume spike
            if (tenkan[i] > kijun[i] and 
                close[i] > cloud_top and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish TK cross (Tenkan < Kijun) AND price below cloud AND below 1d EMA50 AND volume spike
            elif (tenkan[i] < kijun[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below cloud OR bearish TK cross
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above cloud OR bullish TK cross
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals