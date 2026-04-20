#!/usr/bin/env python3
"""
1d_Ichimoku_TF_Trend_Follow_v1
Concept: Ichimoku Cloud trend following on daily timeframe with weekly trend filter.
- Long: Price above Kumo (cloud) AND Tenkan > Kijun AND weekly trend bullish
- Short: Price below Kumo (cloud) AND Tenkan < Kijun AND weekly trend bearish
- Exit: Price crosses opposite Kumo boundary (Senkou Span A or B)
- Uses Kumo twist (Senkou A/B cross) for early trend change detection
- Position sizing: 0.25
- Target: 50-100 total trades over 4 years to balance opportunity and cost
- Ichimoku works in all markets: cloud acts as dynamic S/R, TK cross confirms momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Ichimoku_TF_Trend_Follow_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    # === Daily: Ichimoku Components (9, 26, 52) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # === Weekly: Trend Filter (EMA 21/55 cross) ===
    weekly_close = df_weekly['close'].values
    ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(weekly_close).ewm(span=55, adjust=False, min_periods=55).mean().values
    weekly_bullish = ema21 > ema55  # True when bullish
    weekly_bearish = ema21 < ema55  # True when bearish
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need full Ichimoku calculation
    
    for i in range(start_idx, n):
        # Get values
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(senkou_a_val) or np.isnan(senkou_b_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Kumo boundaries (cloud top and bottom)
        upper_kumo = max(senkou_a_val, senkou_b_val)
        lower_kumo = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: Price above cloud, TK bullish, weekly bullish
            if (close[i] > upper_kumo and 
                tenkan_val > kijun_val and 
                weekly_bull):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, TK bearish, weekly bearish
            elif (close[i] < lower_kumo and 
                  tenkan_val < kijun_val and 
                  weekly_bear):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below cloud bottom OR TK bearish cross
            if (close[i] < lower_kumo or 
                tenkan_val < kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above cloud top OR TK bullish cross
            if (close[i] > upper_kumo or 
                tenkan_val > kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals