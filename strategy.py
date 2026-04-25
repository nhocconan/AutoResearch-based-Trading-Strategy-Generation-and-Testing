#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, Ichimoku TK cross (Tenkan/Kijun) combined with 1d trend filter (price > 1d EMA50) and volume confirmation (>1.5x 20-period average) captures high-probability breakouts in both bull and bear markets. The Kumo (cloud) acts as dynamic support/resistance: long when price is above cloud and TK cross bullish in 1d uptrend; short when price is below cloud and TK cross bearish in 1d downtrend. Exits on opposite TK cross or Kumo penetration. Designed for ~80-120 total trades over 4 years (20-30/year) via tight entry conditions requiring multi-timeframe alignment.
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
    
    # Get 1d data for trend filter and Ichimoku calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need for Ichimoku (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for signals as it requires future data
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # The cloud is between senkou_a and senkou_b
    # We need to align these properly - they are already shifted in calculation
    # But align_htf_to_ltf will handle the timing alignment
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 20, 1)  # Ichimoku needs 52 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get aligned values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Kumo boundaries (cloud top and bottom)
        kumomax = max(senkou_a_val, senkou_b_val)
        kumomin = min(senkou_a_val, senkou_b_val)
        
        # 1d trend filter: price vs EMA50
        # Get 1d close aligned for direct comparison
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_val = close_1d_aligned[i]
        is_uptrend = close_1d_val > ema_50_val
        
        # TK cross signals
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        if position == 0:
            # Look for entry signals
            if is_uptrend:
                # Long conditions: price above cloud, TK bullish cross, volume spike
                long_signal = (close[i] > kumomax) and tk_bullish and vol_spike[i]
            else:
                # Short conditions: price below cloud, TK bearish cross, volume spike
                short_signal = (close[i] < kumomin) and tk_bearish and vol_spike[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. TK cross turns bearish
            # 2. Price penetrates Kumo (cloud)
            if tk_bearish or close[i] < kumomin:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. TK cross turns bullish
            # 2. Price penetrates Kumo (cloud)
            if tk_bullish or close[i] > kumomax:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0