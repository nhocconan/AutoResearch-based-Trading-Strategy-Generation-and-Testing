#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud strategy with 1-week trend filter
# Long when price crosses above Kumo (cloud) AND Tenkan > Kijun AND weekly EMA200 trend up
# Short when price crosses below Kumo AND Tenkan < Kijun AND weekly EMA200 trend down
# Exit when Tenkan crosses back opposite Kijun or price exits cloud in opposite direction
# Uses Ichimoku for momentum and support/resistance, weekly EMA for trend filter
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 for Senkou B)
    start = 52
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long setup: price above cloud AND Tenkan > Kijun AND weekly trend up
            if (price > upper_cloud and tenkan[i] > kijun[i] and price > ema200_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price below cloud AND Tenkan < Kijun AND weekly trend down
            elif (price < lower_cloud and tenkan[i] < kijun[i] and price < ema200_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price drops below cloud
            if tenkan[i] < kijun[i] or price < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price rises above cloud
            if tenkan[i] > kijun[i] or price > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_WeeklyEMA200_Trend"
timeframe = "6h"
leverage = 1.0