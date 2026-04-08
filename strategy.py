#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Weekly Trend Filter + Volume Confirmation
Hypothesis: Ichimoku cloud on 6h with weekly trend filter (price above/below weekly EMA20) and volume confirmation captures strong trends while avoiding false breakouts. The cloud acts as dynamic support/resistance, reducing whipsaws in ranging markets. Weekly trend ensures alignment with higher timeframe momentum. Volume confirms breakout validity. Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = df_1w['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Ichimoku components on 6h
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
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # For signal generation, we use current close vs cloud
    
    # Cloud top and bottom (Senkou Span A and B shifted forward 26 periods)
    # Since we need values available at time i, we use unshifted Senkou spans
    # The cloud ahead is senkou_a and senkou_b (already calculated)
    # For simplicity, we use current Tenkan/Kijun relationship and price vs cloud
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need Senkou B data
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and boundaries
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: price falls below cloud or Tenkan/Kijun cross down
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud or Tenkan/Kijun cross up
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish: price above cloud, Tenkan > Kijun, weekly uptrend, volume
            if (close[i] > cloud_top and 
                tenkan[i] > kijun[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Bearish: price below cloud, Tenkan < Kijun, weekly downtrend, volume
            elif (close[i] < cloud_bottom and 
                  tenkan[i] < kijun[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals