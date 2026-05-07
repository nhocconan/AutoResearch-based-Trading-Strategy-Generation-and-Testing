#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud breakout with 1-day trend filter.
# Long when: Price > Kumo (cloud), Tenkan > Kijun, and 1d trend is up (price > 1d EMA50).
# Short when: Price < Kumo, Tenkan < Kijun, and 1d trend is down (price < 1d EMA50).
# Uses Ichimoku for momentum and support/resistance, with 1d EMA for trend filter.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drift.
# Works in bull markets via upward cloud breaks and in bear via downward breaks.
name = "6h_Ichimoku_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B: (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Sufficient warmup for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Price > cloud, Tenkan > Kijun, and 1d trend up
            long_cond = (close[i] > cloud_top) and (tenkan[i] > kijun[i]) and (close[i] > ema_50_1d_aligned[i])
            # Short: Price < cloud, Tenkan < Kijun, and 1d trend down
            short_cond = (close[i] < cloud_bottom) and (tenkan[i] < kijun[i]) and (close[i] < ema_50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < cloud bottom or Tenkan < Kijun
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > cloud top or Tenkan > Kijun
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals