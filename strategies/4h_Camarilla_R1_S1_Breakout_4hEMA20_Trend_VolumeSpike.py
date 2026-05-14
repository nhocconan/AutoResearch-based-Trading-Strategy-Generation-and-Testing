#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts in direction of 4h EMA20 trend with volume confirmation.
Uses 4h primary timeframe to balance trade frequency and responsiveness. 
Camarilla levels from prior 4h provide structure; 4h EMA20 filters for trend alignment; 
volume spike confirms breakout. Works in bull/bear via trend filter + volume confirmation.
Target: 20-50 trades/year per symbol to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Camarilla levels for today (based on prior 4h OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_range = (high_4h - low_4h) * 1.1 / 12.0
    camarilla_R1 = close_4h + camarilla_range
    camarilla_S1 = close_4h - camarilla_range
    
    # Align Camarilla levels to 4h timeframe (prior 4h's levels available at 4h close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1)
    
    # Volume confirmation: volume > 2.0x 20-period average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20 (20), volume MA (20), aligned indicators
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 4h EMA20
        price_above_ema = close[i] > ema_20_4h_aligned[i]
        price_below_ema = close[i] < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 4h EMA20 + volume spike
            long_breakout = close[i] > camarilla_R1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i]
            
            # Short: price breaks below Camarilla S1 + price below 4h EMA20 + volume spike
            short_breakout = close[i] < camarilla_S1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 OR trend turns bearish (price below EMA)
            if (close[i] < camarilla_S1_aligned[i] or not price_above_ema):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish (price above EMA)
            if (close[i] > camarilla_R1_aligned[i] or not price_below_ema):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0