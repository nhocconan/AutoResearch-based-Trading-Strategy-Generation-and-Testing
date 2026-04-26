#!/usr/bin/env python3
"""
1d_Ichimoku_Cloud_Trend_WeeklyFilter_v1
Hypothesis: 1d Ichimoku cloud strategy with weekly trend filter for BTC/ETH.
- Uses 1d timeframe for lower trade frequency (target: 30-100 total trades over 4 years)
- Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 1d data
- Weekly EMA200 filter ensures trades align with higher timeframe trend (bull/bear agnostic)
- Long when price > cloud AND Tenkan > Kijun AND weekly trend up
- Short when price < cloud AND Tenkan < Kijun AND weekly trend down
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the weekly trend and using Ichimoku for entry timing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Ichimoku calculations (9, 26, 52 periods)
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
    
    # Align Ichimoku components to current timeframe (no shift needed as they're based on completed periods)
    tenkan_aligned = tenkan  # Already calculated from historical data
    kijun_aligned = kijun
    senkou_a_aligned = senkou_a
    senkou_b_aligned = senkou_b
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        price_above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun AND weekly uptrend
            if price_above_cloud and tenkan_above_kijun and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND Tenkan < Kijun AND weekly downtrend
            elif price_below_cloud and tenkan_below_kijun and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR Tenkan crosses below Kijun
            if price_below_cloud or tenkan_below_kijun:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR Tenkan crosses above Kijun
            if price_above_cloud or tenkan_above_kijun:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Ichimoku_Cloud_Trend_WeeklyFilter_v1"
timeframe = "1d"
leverage = 1.0