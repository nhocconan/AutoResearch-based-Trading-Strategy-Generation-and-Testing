#!/usr/bin/env python3
"""
6H_Ichimoku_Cloud_Breakout_With_1dTrend_Filter
Hypothesis: Ichimoku cloud provides dynamic support/resistance and trend direction.
In strong trends (price above/below cloud), breakouts in direction of 1d trend have high probability.
Combines TK cross for entry timing with cloud filter to avoid false breakouts.
Designed for ~20-40 trades/year on 6f timeframe to minimize fee drag.
Works in both bull/bear markets by aligning with higher timeframe trend.
"""

name = "6H_Ichimoku_Cloud_Breakout_With_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # === Ichimoku Components (9, 26, 52) ===
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used in signals to avoid look-ahead
    
    # === Cloud Top and Bottom ===
    # Cloud top is the higher of Senkou A and B
    # Cloud bottom is the lower of Senkou A and B
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # === Price Position Relative to Cloud ===
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    price_in_cloud = (close >= cloud_bottom) & (close <= cloud_top)
    
    # === TK Cross (Tenkan/Kijun Cross) ===
    # Bullish cross: Tenkan crosses above Kijun
    # Bearish cross: Tenkan crosses below Kijun
    tk_bullish = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_bearish = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    # Handle first element
    tk_bullish[0] = False
    tk_bearish[0] = False
    
    # === Load Daily Trend Filter (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Ichimoku calculations)
    start_idx = 52  # covers the longest period (52) in Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d trend direction
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Price above cloud + TK bullish cross + 1d uptrend + volume
            if price_above_cloud[i] and tk_bullish[i] and trend_up and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price below cloud + TK bearish cross + 1d downtrend + volume
            elif price_below_cloud[i] and tk_bearish[i] and trend_down and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below cloud OR TK bearish cross
                if close[i] < cloud_top[i] or tk_bearish[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above cloud OR TK bullish cross
                if close[i] > cloud_bottom[i] or tk_bullish[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals