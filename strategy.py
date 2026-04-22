#!/usr/bin/env python3

"""
Hypothesis: 6-hour Ichimoku Cloud Breakout with 1-week Trend Filter and Volume Confirmation.
Trades breakouts above/below the 1-week Ichimoku cloud in the direction of the weekly trend (Tenkan > Kijun).
Uses volume spike to confirm institutional interest. Designed for low trade frequency (12-37/year) to minimize
fee drift and work in both bull and bear markets by aligning with higher timeframe trend and using cloud as
dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components for given high, low, close arrays."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Ichimoku and trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Weekly Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Weekly trend: Tenkan > Kijun = uptrend, Tenkan < Kijun = downtrend
    weekly_trend = tenkan_1w - kijun_1w  # positive = uptrend, negative = downtrend
    
    # Cloud boundaries (Senkou Span A and B)
    # Note: Ichimoku cloud is plotted 26 periods ahead, so we need to shift back for current period
    senkou_a_shifted = np.roll(senkou_a_1w, 26)
    senkou_b_shifted = np.roll(senkou_b_1w, 26)
    # First 26 values will be invalid due to roll, but we'll handle with checks
    
    # Align to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_shifted)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready (check for NaN from rolling or roll)
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Cloud boundaries: top = max(senkou_a, senkou_b), bottom = min(senkou_a, senkou_b)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0 and vol_spike:
            # Long: price breaks above cloud with bullish weekly trend
            if close[i] > cloud_top and weekly_trend_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with bearish weekly trend
            elif close[i] < cloud_bottom and weekly_trend_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite cloud boundary or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below cloud bottom or weekly trend turns bearish
                if close[i] < cloud_bottom or weekly_trend_aligned[i] < 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above cloud top or weekly trend turns bullish
                if close[i] > cloud_top or weekly_trend_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0