#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud Breakout with Weekly Trend Filter
# Hypothesis: Ichimoku cloud acts as dynamic support/resistance, with weekly trend
# filtering to avoid counter-trend trades. Works in bull/bear by capturing
# breakouts in trending markets and avoiding whipsaws in ranging markets.
# Target: 12-37 trades/year (48-148 over 4 years).

name = "6h_ichimoku_cloud_breakout_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Calculate weekly EMA26 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema26 = pd.Series(weekly_close).ewm(span=26, adjust=False).mean().values
    weekly_ema26_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema26)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to current timeframe
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(weekly_ema26_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine trend from weekly EMA26
        weekly_trend = 1 if close[i] > weekly_ema26_aligned[i] else -1
        
        if position == 1:  # Long position
            # Exit: price closes below cloud bottom (cloud break) or opposite TK cross
            if close[i] < cloud_bottom or (tenkan_aligned[i] < kijun_aligned[i] and weekly_trend == -1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above cloud top (cloud break) or opposite TK cross
            if close[i] > cloud_top or (tenkan_aligned[i] > kijun_aligned[i] and weekly_trend == 1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # TK cross with weekly trend filter
            # Bullish TK cross: Tenkan crosses above Kijun
            # Bearish TK cross: Tenkan crosses below Kijun
            bullish_cross = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            bearish_cross = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            # Enter long: bullish TK cross above cloud in uptrend
            if bullish_cross and close[i] > cloud_top and weekly_trend == 1:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish TK cross below cloud in downtrend
            elif bearish_cross and close[i] < cloud_bottom and weekly_trend == -1:
                position = -1
                signals[i] = -0.25
    
    return signals