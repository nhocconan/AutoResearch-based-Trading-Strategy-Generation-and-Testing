#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud (10,26,52) with weekly trend filter and volume confirmation.
- Ichimoku Cloud (Tenkan/Kijun cross + price vs cloud) provides trend direction and support/resistance
- Weekly EMA200 filter ensures alignment with long-term trend, reducing counter-trend trades
- Volume spike (1.5x 20-period average) confirms institutional participation
- Target: 20-40 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
- Works in bull/bear: Cloud acts as dynamic support/resistance, weekly filter avoids counter-trend
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).values
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Ichimoku components (9,26,52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 8:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
        else:
            period9_high[i] = np.max(high[:i+1]) if i >= 0 else np.nan
            period9_low[i] = np.min(low[:i+1]) if i >= 0 else np.nan
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 25:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
        else:
            period26_high[i] = np.max(high[:i+1]) if i >= 0 else np.nan
            period26_low[i] = np.min(low[:i+1]) if i >= 0 else np.nan
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 51:
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
        else:
            period52_high[i] = np.max(high[:i+1]) if i >= 0 else np.nan
            period52_low[i] = np.min(low[:i+1]) if i >= 0 else np.nan
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Ichimoku cloud (Senkou Span A/B from 26 periods ago)
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(52, 100)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above cloud + Tenkan > Kijun + weekly uptrend + volume spike
            if (close[i] > cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud + Tenkan < Kijun + weekly downtrend + volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below cloud OR Tenkan < Kijun (trend weakening)
            if (close[i] < cloud_bottom[i] or 
                tenkan[i] < kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above cloud OR Tenkan > Kijun (trend weakening)
            if (close[i] > cloud_top[i] or 
                tenkan[i] > kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_WeeklyEMA200_Volume_v1"
timeframe = "6h"
leverage = 1.0