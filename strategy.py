#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Ichimoku trend (cloud color) and EMA50 filter.
- Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displaced).
- Trend filter: Price above/below 1d EMA50 to align with higher timeframe trend.
- Entry: Long when Tenkan crosses above Kijun AND price is above cloud (Senkou Span A) AND close > 1d EMA50 with volume spike.
         Short when Tenkan crosses below Kijun AND price is below cloud (Senkou Span B) AND close < 1d EMA50 with volume spike.
- Exit: When Tenkan crosses back in opposite direction (Kijun cross) or price re-enters the cloud.
- Uses discrete signal size: 0.25 to manage drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying bullish TK crosses in uptrend, in bear via selling bearish TK crosses in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align 1d indicators to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need 52 periods for Senkou B + 26 displacement = 78, but align_htf_to_ltf handles displacement
    # Actually need enough for Ichimoku calculation: max(52, 26, 9) = 52
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        # Cloud top is max(Senkou A, Senkou B), cloud bottom is min(Senkou A, Senkou B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Check for TK cross signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish TK cross: Tenkan crosses above Kijun
                bullish_tk_cross = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
                # Bearish TK cross: Tenkan crosses below Kijun
                bearish_tk_cross = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
                
                if bullish_tk_cross:
                    # Long when price is above cloud AND close > 1d EMA50 (uptrend)
                    if close[i] > cloud_top and close[i] > ema_50_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif bearish_tk_cross:
                    # Short when price is below cloud AND close < 1d EMA50 (downtrend)
                    if close[i] < cloud_bottom and close[i] < ema_50_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun (TK cross reversal) OR price re-enters cloud
            bearish_tk_cross = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            if bearish_tk_cross or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun (TK cross reversal) OR price re-enters cloud
            bullish_tk_cross = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            if bullish_tk_cross or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0