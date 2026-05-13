#!/usr/bin/env python3
# 4h_Ichimoku_Cloud_Twist_Perfect_Order_Volume
# Hypothesis: Ichimoku cloud twist (Tenkan/Kijun cross) with perfect order alignment (price above/below all Ichimoku lines) and volume confirmation captures strong trends with controlled frequency.
# Works in bull markets via bullish twist + price above cloud + volume spike; in bear markets via bearish twist + price below cloud + volume spike.
# Uses cloud twist as primary signal, perfect order for trend strength, and volume for confirmation to reduce false signals.
# Target: 25-40 trades per year per symbol to minimize fee drag.

name = "4h_Ichimoku_Cloud_Twist_Perfect_Order_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Cloud twist signals
    # Bullish twist: Tenkan crosses above Kijun
    # Bearish twist: Tenkan crosses below Kijun
    bullish_twist = (tenkan > kijun) & (tenkan <= kijun)  # Will be handled in loop with previous values
    bearish_twist = (tenkan < kijun) & (tenkan >= kijun)  # Will be handled in loop with previous values
    
    # Perfect order conditions
    # Bullish perfect order: price > Tenkan > Kijun > Senkou A > Senkou B
    # Bearish perfect order: price < Tenkan < Kijun < Senkou A < Senkou B
    
    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):
        # Skip if any required value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Check for Ichimoku twist with perfect order and volume confirmation
            # Bullish setup: Tenkan crossed above Kijun (twist) + perfect order + volume
            bullish_twist_now = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            bullish_perfect_order = (close[i] > tenkan[i] and 
                                   tenkan[i] > kijun[i] and 
                                   kijun[i] > senkou_a[i] and 
                                   senkou_a[i] > senkou_b[i])
            
            # Bearish setup: Tenkan crossed below Kijun (twist) + perfect order + volume
            bearish_twist_now = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            bearish_perfect_order = (close[i] < tenkan[i] and 
                                   tenkan[i] < kijun[i] and 
                                   kijun[i] < senkou_a[i] and 
                                   senkou_a[i] < senkou_b[i])
            
            if bullish_twist_now and bullish_perfect_order and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            elif bearish_twist_now and bearish_perfect_order and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun (twist reversal) or perfect order breaks
            bearish_twist_now = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            perfect_order_broken = not (close[i] > tenkan[i] and 
                                      tenkan[i] > kijun[i] and 
                                      kijun[i] > senkou_a[i] and 
                                      senkou_a[i] > senkou_b[i])
            
            if bearish_twist_now or perfect_order_broken:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun (twist reversal) or perfect order breaks
            bullish_twist_now = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            perfect_order_broken = not (close[i] < tenkan[i] and 
                                      tenkan[i] < kijun[i] and 
                                      kijun[i] < senkou_a[i] and 
                                      senkou_a[i] < senkou_b[i])
            
            if bullish_twist_now or perfect_order_broken:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals