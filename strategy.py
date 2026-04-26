#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: 6h Ichimoku TK cross (Tenkan/Kijun) with 1d cloud filter and volume confirmation.
Long when TK crosses bullish above cloud in 1d uptrend with volume spike. Short when TK crosses bearish below cloud in 1d downtrend with volume spike.
Uses Ichimoku's built-in trend/cloud filter to avoid whipsaws in ranging markets.
Designed for fewer, higher-quality trades (target: 12-37/year) with discrete sizing (0.25).
Works in both bull and bear markets by using 1d cloud as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need 52 for Senkou Span B
        return np.zeros(n)
    
    # Get 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (they are already on 6h)
    # But we need to shift Senkou spans forward by 26 periods (they are plotted 26 periods ahead)
    # For trading, we use current Senkou spans (which were calculated 26 periods ago)
    # So we don't shift - we use the values as calculated
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Get 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.8x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish_cross = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish_cross = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            
            # Long: bullish TK cross above cloud with 1d uptrend and volume spike
            if (tk_bullish_cross and 
                close[i] > cloud_top[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross below cloud with 1d downtrend and volume spike
            elif (tk_bearish_cross and 
                  close[i] < cloud_bottom[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross turns bearish OR price drops below cloud bottom
            tk_bearish_cross = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            if tk_bearish_cross or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross turns bullish OR price rises above cloud top
            tk_bullish_cross = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            if tk_bullish_cross or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0