#!/usr/bin/env python3
name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1d Data for Ichimoku calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Ichimoku Components (9, 26, 52) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal generation, we use current price vs cloud
    
    # Align Ichimoku components to 6h timeframe (using previous day's values)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === Kumo Twist Detection ===
    # Kumo twist occurs when Senkou A and Senkou B cross
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    kumo_twist_bull = senkou_a_6h > senkou_b_6h
    kumo_twist_bear = senkou_a_6h < senkou_b_6h
    
    # === TK Cross (Tenkan/Kijun crossover) ===
    # Bullish TK: Tenkan crosses above Kijun
    # Bearish TK: Tenkan crosses below Kijun
    tk_cross_bull = tenkan_6h > kijun_6h
    tk_cross_bear = tenkan_6h < kijun_6h
    
    # === Price vs Cloud ===
    # Price above cloud: bullish
    # Price below cloud: bearish
    price_above_cloud = close > np.maximum(senkou_a_6h, senkou_b_6h)
    price_below_cloud = close < np.minimum(senkou_a_6h, senkou_b_6h)
    
    # === 1d Trend Filter (EMA 50) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    price_above_ema = close > ema50_6h
    price_below_ema = close < ema50_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50)  # Need enough data for Ichimoku and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or
            np.isnan(senkou_b_6h[i]) or
            np.isnan(ema50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish TK cross + price above cloud + above 1d EMA50
            if (tk_cross_bull[i] and 
                price_above_cloud[i] and
                price_above_ema[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross + price below cloud + below 1d EMA50
            elif (tk_cross_bear[i] and 
                  price_below_cloud[i] and
                  price_below_ema[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish TK cross or price below cloud
            if tk_cross_bear[i] or price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish TK cross or price above cloud
            if tk_cross_bull[i] or price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals