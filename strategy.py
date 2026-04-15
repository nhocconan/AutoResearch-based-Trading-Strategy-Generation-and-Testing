#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Filter with Weekly Trend and Volume Confirmation
# Uses Ichimoku cloud from daily timeframe for trend filtering (price above/below cloud),
# Tenkan-Kijun cross from 6h for entry timing, and weekly trend direction from 1w.
# Works in bull markets (long when price above cloud + bullish TK cross + weekly up)
# and bear markets (short when price below cloud + bearish TK cross + weekly down).
# Volume confirmation reduces false signals. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Weekly trend: 20-period EMA of weekly close
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = close_1w > ema20_1w
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(weekly_trend_aligned[i])):
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Volume confirmation: current volume > 1.5x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_confirm = volume[i] > 1.5 * vol_median
        
        # Long entry: price above cloud + bullish TK cross + weekly up + volume
        if (close[i] > cloud_top and
            tk_cross_bullish and
            weekly_trend_aligned[i] > 0.5 and
            volume_confirm and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below cloud + bearish TK cross + weekly down + volume
        elif (close[i] < cloud_bottom and
              tk_cross_bearish and
              weekly_trend_aligned[i] < 0.5 and
              volume_confirm and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite TK cross or price returns to cloud
        elif position == 1 and (tk_cross_bearish or close[i] < cloud_top):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_cross_bullish or close[i] > cloud_bottom):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0