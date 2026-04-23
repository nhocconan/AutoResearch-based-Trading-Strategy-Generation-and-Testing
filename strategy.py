#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + 1d EMA50 Trend + Volume Spike Confirmation
- Uses Ichimoku system (Tenkan-sen, Kijun-sen, Senkou Span A/B) for trend and momentum
- Only take trades in direction of 1d EMA50 trend to avoid counter-trend whipsaws
- Requires volume > 1.8x 20-period average for confirmation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Ichimoku cloud acts as dynamic support/resistance, reducing false breakouts
- Works in both bull and bear markets by aligning with higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current Ichimoku cloud (already shifted in calculation)
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20, 50)  # Ichimoku, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Ichimoku signals
        # Bullish: Tenkan-sen crosses above Kijun-sen AND price above cloud
        tk_cross_bull = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        price_above_cloud = close[i] > upper_cloud[i]
        
        # Bearish: Tenkan-sen crosses below Kijun-sen AND price below cloud
        tk_cross_bear = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        price_below_cloud = close[i] < lower_cloud[i]
        
        if position == 0:
            # Long conditions: bullish TK cross, price above cloud, uptrend, volume spike
            long_signal = (tk_cross_bull and 
                          price_above_cloud and
                          uptrend and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: bearish TK cross, price below cloud, downtrend, volume spike
            short_signal = (tk_cross_bear and 
                           price_below_cloud and
                           downtrend and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: TK cross in opposite direction or price crosses cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish TK cross OR price falls below cloud
                if (tk_cross_bear or 
                    close[i] < lower_cloud[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish TK cross OR price rises above cloud
                if (tk_cross_bull or 
                    close[i] > upper_cloud[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0