#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation
# Uses Ichimoku components from prior completed 1d bar for structure (Tenkan-sen/Kijun-sen cross + price above/below cloud)
# 1w EMA50 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in both bull (cloud breakout continuation) and bear (cloud breakdown continuation) markets
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias, more robust across regimes)

name = "6h_Ichimoku_Cloud_1wEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components from prior completed 1d bar
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    tenkan_sen_shifted = np.roll(tenkan_sen, 1)
    kijun_sen_shifted = np.roll(kijun_sen, 1)
    senkou_span_a_shifted = np.roll(senkou_span_a, 1)
    senkou_span_b_shifted = np.roll(senkou_span_b, 1)
    tenkan_sen_shifted[0] = np.nan
    kijun_sen_shifted[0] = np.nan
    senkou_span_a_shifted[0] = np.nan
    senkou_span_b_shifted[0] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_shifted)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_shifted)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long conditions: Tenkan > Kijun AND price above cloud AND price > 1w EMA50 AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > cloud_top and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > (1.8 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Tenkan < Kijun AND price below cloud AND price < 1w EMA50 AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > (1.8 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan < Kijun OR price crosses below cloud bottom
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan > Kijun OR price crosses above cloud top
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals