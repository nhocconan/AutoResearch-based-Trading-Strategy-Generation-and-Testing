#!/usr/bin/env python3

"""
Hypothesis: 6h Ichimoku Cloud (Tenkan/Kijun + Senkou Span) with weekly trend filter.
Go long when Tenkan crosses above Kijun, price is above cloud, and weekly trend is up.
Go short when Tenkan crosses below Kijun, price is below cloud, and weekly trend is down.
Ichimoku provides multi-factor confirmation (momentum, trend, support/resistance) in one system.
Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws.
Designed for low trade frequency (12-37/year) by requiring multiple confirmations.
Works in both bull and bear markets by following the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52)
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
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (shifted back to align with current price)
    # Senkou Span values are plotted 26 periods ahead, so to get current cloud we look back 26
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll, but we start loop after warmup anyway
    
    # Upper cloud (Kumo) is the higher of Senkou A and Senkou B
    upper_cloud = np.maximum(senkou_a_shifted, senkou_b_shifted)
    # Lower cloud is the lower of Senkou A and Senkou B
    lower_cloud = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(ema50_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku signals
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        tk_cross_up = tenkan[i] > kijun[i] and tenkan_prev <= kijun_prev
        tk_cross_down = tenkan[i] < kijun[i] and tenkan_prev >= kijun_prev
        
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        
        # Weekly trend: slope of EMA50
        weekly_up = ema50_weekly_aligned[i] > ema50_weekly_aligned[i-1]
        weekly_down = ema50_weekly_aligned[i] < ema50_weekly_aligned[i-1]
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up and price_above_cloud and weekly_up:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down and price_below_cloud and weekly_down:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross in opposite direction or price enters cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: TK cross down or price drops below cloud
                if tk_cross_down or close[i] < upper_cloud[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: TK cross up or price rises above cloud
                if tk_cross_up or close[i] > lower_cloud[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Ichimoku_Cloud_WeeklyTrend_6h"
timeframe = "6h"
leverage = 1.0