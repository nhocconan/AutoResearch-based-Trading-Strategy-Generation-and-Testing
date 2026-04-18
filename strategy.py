#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend
Ichimoku strategy on 6h with weekly trend filter:
- Long when Tenkan-sen > Kijun-sen (TK cross up) AND price above cloud
- Short when Tenkan-sen < Kijun-sen (TK cross down) AND price below cloud
- Weekly trend filter: only long when price > weekly EMA34, short when price < weekly EMA34
- Designed for 15-25 trades/year per symbol (~60-100 total over 4 years)
Works in trending markets (TK cross with trend) and avoids false signals in ranging markets (cloud filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components: Tenkan-sen, Kijun-sen, Senkou Span A/B."""
    n = len(high)
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Ichimoku components
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need 52 for Senkou B + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # TK cross conditions
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up and price_above_cloud and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down and price_below_cloud and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down OR price below cloud OR weekly trend turns down
            if tk_cross_down or close[i] < cloud_top or close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up OR price above cloud OR weekly trend turns up
            if tk_cross_up or close[i] > cloud_bottom or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0