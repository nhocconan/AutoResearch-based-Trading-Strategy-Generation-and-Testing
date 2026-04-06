#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Tenkan/Kijun cross and Kumo twist filter
# Long when Tenkan crosses above Kijun AND price above Kumo AND Kumo twisting bullish (Senkou A rising)
# Short when Tenkan crosses below Kijun AND price below Kumo AND Kumo twisting bearish (Senkou A falling)
# Exit when price crosses Tenkan-Kijun average (Kijun-sen) or Kumo twist changes
# Uses Ichimoku's inherent trend/momentum with daily timeframe for structure
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_ichimoku_1d_kumo_twist_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).values
    
    # Kumo twist: Senkou A slope (current - previous)
    senkou_a_slope = np.diff(senkou_a, prepend=senkou_a[0])
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal generation, we use current price vs Kumo (cloud)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(senkou_a_slope[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Kumo (cloud) boundaries
        kumO_top = np.maximum(senkou_a[i], senkou_b[i])
        kumO_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # Check exits: price crosses Kijun-sen OR Kumo twist changes direction
        if position == 1:  # long position
            if (close[i] < kijun[i] or 
                (i > 0 and senkou_a_slope[i] * senkou_a_slope[i-1] < 0)):  # Kumo twist changed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (close[i] > kijun[i] or 
                (i > 0 and senkou_a_slope[i] * senkou_a_slope[i-1] < 0)):  # Kumo twist changed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Ichimoku signals
            # Bullish: Tenkan crosses above Kijun AND price above Kumo AND Kumo twisting bullish
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # Tenkan/Kijun cross up
                close[i] > kumO_top and  # Price above cloud
                senkou_a_slope[i] > 0):  # Kumo twisting bullish (Senkou A rising)
                signals[i] = 0.25
                position = 1
            # Bearish: Tenkan crosses below Kijun AND price below Kumo AND Kumo twisting bearish
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # Tenkan/Kijun cross down
                  close[i] < kumO_bottom and  # Price below cloud
                  senkou_a_slope[i] < 0):  # Kumo twisting bearish (Senkou A falling)
                signals[i] = -0.25
                position = -1
    
    return signals