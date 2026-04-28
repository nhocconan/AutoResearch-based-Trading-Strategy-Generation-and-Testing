#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_WeeklyTrend
Hypothesis: Uses Ichimoku cloud twist (Senkou Span A/B crossover) on weekly timeframe as trend filter,
with Tenkan/Kijun cross on daily timeframe for entry timing on 6-hour chart.
Adds volume confirmation to filter false signals.
Designed to work in both bull and bear markets by using cloud twist for trend direction
and TK cross for momentum, avoiding whipsaws in ranging markets.
Targets 12-37 trades per year to minimize fee drag while capturing meaningful momentum shifts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals but calculated for completeness
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku trend filter (cloud twist)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 60:
        return np.zeros(n)
    
    # Get daily data for TK cross entry signal
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku on weekly data
    tenkan_w, kijun_w, senkou_a_w, senkou_b_w = calculate_ichimoku(
        df_weekly['high'].values,
        df_weekly['low'].values,
        df_weekly['close'].values
    )
    
    # Cloud twist: Senkou A/B crossover indicates trend change
    # When Senkou A crosses above Senkou B = bullish twist
    # When Senkou A crosses below Senkou B = bearish twist
    senkou_a_w_shift = np.roll(senkou_a_w, 1)
    senkou_b_w_shift = np.roll(senkou_b_w, 1)
    senkou_a_w_shift[0] = np.nan
    senkou_b_w_shift[0] = np.nan
    
    # Bullish twist: Senkou A crosses above Senkou B
    bullish_twist = (senkou_a_w > senkou_b_w) & (senkou_a_w_shift <= senkou_b_w_shift)
    # Bearish twist: Senkou A crosses below Senkou B
    bearish_twist = (senkou_a_w < senkou_b_w) & (senkou_a_w_shift >= senkou_b_w_shift)
    
    # Cumulative twist state for trend filter
    bullish_trend = np.zeros_like(tenkan_w, dtype=bool)
    bearish_trend = np.zeros_like(tenkan_w, dtype=bool)
    
    for i in range(1, len(tenkan_w)):
        if bullish_twist[i]:
            bullish_trend[i] = True
            bearish_trend[i] = False
        elif bearish_twist[i]:
            bullish_trend[i] = False
            bearish_trend[i] = True
        else:
            bullish_trend[i] = bullish_trend[i-1]
            bearish_trend[i] = bearish_trend[i-1]
    
    # Align weekly trend to 6h timeframe
    bullish_trend_aligned = align_htf_to_ltf(prices, df_weekly, bullish_trend.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_weekly, bearish_trend.astype(float))
    
    # Calculate Ichimoku on daily data for TK cross entry
    tenkan_d, kijun_d, senkou_a_d, senkou_b_d = calculate_ichimoku(
        df_daily['high'].values,
        df_daily['low'].values,
        df_daily['close'].values
    )
    
    # TK cross: Tenkan crosses Kijun
    tenkan_d_shift = np.roll(tenkan_d, 1)
    kijun_d_shift = np.roll(kijun_d, 1)
    tenkan_d_shift[0] = np.nan
    kijun_d_shift[0] = np.nan
    
    # Bullish TK cross: Tenkan crosses above Kijun
    tk_bullish = (tenkan_d > kijun_d) & (tenkan_d_shift <= kijun_d_shift)
    # Bearish TK cross: Tenkan crosses below Kijun
    tk_bearish = (tenkan_d < kijun_d) & (tenkan_d_shift >= kijun_d_shift)
    
    # Align daily TK cross to 6h timeframe
    tk_bullish_aligned = align_htf_to_ltf(prices, df_daily, tk_bullish.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_daily, tk_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from weekly Ichimoku cloud twist
        trend_up = bullish_trend_aligned[i] > 0.5
        trend_down = bearish_trend_aligned[i] > 0.5
        
        # Entry signals from daily TK cross
        tk_cross_up = tk_bullish_aligned[i] > 0.5
        tk_cross_down = tk_bearish_aligned[i] > 0.5
        
        # Volume confirmation
        vol_ok = vol_confirm[i]
        
        # Entry logic:
        # Long: Bullish TK cross in bullish weekly trend
        long_entry = tk_cross_up and trend_up and vol_ok
        # Short: Bearish TK cross in bearish weekly trend
        short_entry = tk_cross_down and trend_down and vol_ok
        
        # Exit logic: Opposite TK cross or trend change
        long_exit = tk_cross_down or not trend_up
        short_exit = tk_cross_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_Twist_WeeklyTrend"
timeframe = "6h"
leverage = 1.0