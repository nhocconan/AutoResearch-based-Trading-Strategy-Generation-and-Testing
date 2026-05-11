#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_1dTrend
Hypothesis: Uses Ichimoku cloud twist (Tenkan/Kijun cross) with cloud thickness filter and 1-day trend filter.
Trades in direction of daily trend only when cloud twist occurs and cloud is thin (low volatility).
Designed for low trade frequency (<25/year) to avoid fee drift while capturing high-probability trend continuations.
"""

name = "6h_Ichimoku_Cloud_Twist_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Ichimoku Components (9, 26, 52) ---
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(52)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used for signals)
    
    # Cloud thickness: absolute difference between Senkou spans
    cloud_thickness = np.abs(senkou_a - senkou_b)
    
    # Cloud twist signals: Tenkan/Kijun cross
    # Bullish twist: Tenkan crosses above Kijun
    # Bearish twist: Tenkan crosses below Kijun
    bullish_twist = (tenkan.shift(1) <= kijun.shift(1)) & (tenkan > kijun)
    bearish_twist = (tenkan.shift(1) >= kijun.shift(1)) & (tenkan < kijun)
    
    # Cloud is thin (low volatility) - using 20-period average of thickness
    cloud_thick_ma = pd.Series(cloud_thickness).rolling(window=20, min_periods=20).mean()
    thin_cloud = cloud_thickness < cloud_thick_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max lookback is 52 for Senkou B)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan.iloc[i]) or
            np.isnan(kijun.iloc[i]) or
            np.isnan(senkou_a.iloc[i]) or
            np.isnan(senkou_b.iloc[i]) or
            np.isnan(thin_cloud.iloc[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            if price_above_ema:
                # Uptrend: look for bullish twist with thin cloud
                if bullish_twist.iloc[i] and thin_cloud.iloc[i]:
                    signals[i] = 0.25
                    position = 1
            elif price_below_ema:
                # Downtrend: look for bearish twist with thin cloud
                if bearish_twist.iloc[i] and thin_cloud.iloc[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: bearish twist or price closes below Kijun
                exit_signal = bearish_twist.iloc[i] or (close[i] < kijun.iloc[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish twist or price closes above Kijun
                exit_signal = bullish_twist.iloc[i] or (close[i] > kijun.iloc[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals