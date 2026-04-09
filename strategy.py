#!/usr/bin/env python3
# 6h_ichimoku_trend_follow_v1
# Hypothesis: 6h Ichimoku cloud strategy with 1d HTF trend filter. Enters long when price is above cloud, Tenkan > Kijun, and bullish 1d EMA50; short when price below cloud, Tenkan < Kijun, and bearish 1d EMA50. Uses discrete sizing (0.25) to limit fee drag. Designed for 6h timeframe to capture medium-term trends while avoiding whipsaws via cloud filter and HTF alignment. Targets 12-37 trades/year by requiring confluence of multiple trend-following conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    # At time t, cloud is Senkou from t-26
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # 1d HTF trend filter: 50-period EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(close[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below cloud or Tenkan < Kijun
            if close[i] < cloud_bottom[i] or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud or Tenkan > Kijun
            if close[i] > cloud_top[i] or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with Ichimoku signals and 1d trend alignment
            # Bullish: price above cloud, Tenkan > Kijun
            bullish_ichimoku = (close[i] > cloud_top[i]) and (tenkan[i] > kijun[i])
            # Bearish: price below cloud, Tenkan < Kijun
            bearish_ichimoku = (close[i] < cloud_bottom[i]) and (tenkan[i] < kijun[i])
            
            # 1d trend filter
            bullish_trend = close[i] > ema_50_1d_aligned[i]
            bearish_trend = close[i] < ema_50_1d_aligned[i]
            
            # Long: bullish Ichimoku + bullish 1d trend
            if bullish_ichimoku and bullish_trend:
                position = 1
                signals[i] = 0.25
            # Short: bearish Ichimoku + bearish 1d trend
            elif bearish_ichimoku and bearish_trend:
                position = -1
                signals[i] = -0.25
    
    return signals