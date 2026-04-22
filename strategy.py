# Solution
#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter.
Long when price is above cloud, Tenkan-sen > Kijun-sen, and 1-day EMA50 rising.
Short when price is below cloud, Tenkan-sen < Kijun-sen, and 1-day EMA50 falling.
Exit when price crosses opposite cloud boundary or Tenkan/Kijun cross reverses.
Ichimoku provides dynamic support/resistance; 1-day EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 6h Ichimoku for entries.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Cloud top and bottom (Senkou Span A and B shifted forward 26 periods)
    # For signal at index i, we use Senkou values from i-26 (already shifted in data)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll, handled by nan checks
    
    # Cloud boundaries: top = max(senkou_a, senkou_b), bottom = min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after enough data for Ichimoku
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun, and 1-day EMA50 rising
            if (close[i] > cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun, and 1-day EMA50 falling
            elif (close[i] < cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below cloud OR Tenkan crosses below Kijun
                if (close[i] < cloud_bottom[i] or 
                    tenkan[i] < kijun[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above cloud OR Tenkan crosses above Kijun
                if (close[i] > cloud_top[i] or 
                    tenkan[i] > kijun[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0