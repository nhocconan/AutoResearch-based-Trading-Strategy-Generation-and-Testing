#!/usr/bin/env python3
# 6H_Aroon_Cross_1dTrend_VolumeBreakout
# Hypothesis: Uses Aroon oscillator (25-period) on 6h timeframe to detect trend strength and direction.
# Long when Aroon Up crosses above Aroon Down (bullish trend) with volume breakout (>2x 20-period avg) and 1d uptrend (close > EMA50).
# Short when Aroon Down crosses above Aroon Up (bearish trend) with volume breakout and 1d downtrend (close < EMA50).
# Exits when Aroon cross reverses or volume drops below average.
# Designed to work in both bull/bear markets by using 1d trend filter and volume confirmation to avoid whipsaws.
# Targets 12-30 trades per year on 6h timeframe with position size 0.25.

name = "6H_Aroon_Cross_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Aroon Oscillator (25-period) on 6h data
    # Aroon Up = ((period - periods since highest high) / period) * 100
    # Aroon Down = ((period - periods since lowest low) / period) * 100
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period - 1, n):
        window_high = high[i - aroon_period + 1:i + 1]
        window_low = low[i - aroon_period + 1:i + 1]
        highest_high_idx = np.argmax(window_high)
        lowest_low_idx = np.argmin(window_low)
        periods_since_high = aroon_period - 1 - highest_high_idx
        periods_since_low = aroon_period - 1 - lowest_low_idx
        aroon_up[i] = ((aroon_period - periods_since_high) / aroon_period) * 100
        aroon_down[i] = ((aroon_period - periods_since_low) / aroon_period) * 100
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_breakout = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, aroon_period - 1, 20)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Aroon cross signals
        aroon_up_cross = aroon_up[i] > aroon_down[i] and aroon_up[i-1] <= aroon_down[i-1]
        aroon_down_cross = aroon_down[i] > aroon_up[i] and aroon_down[i-1] <= aroon_up[i-1]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Aroon Up crosses above Aroon Down with volume breakout and uptrend
            if aroon_up_cross and volume_breakout[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: Aroon Down crosses above Aroon Up with volume breakout and downtrend
            elif aroon_down_cross and volume_breakout[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Aroon Down crosses above Aroon Up or volume drops below average
            if aroon_down_cross or not volume_breakout[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Aroon Up crosses above Aroon Down or volume drops below average
            if aroon_up_cross or not volume_breakout[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals