#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_v1
Hypothesis: On 6h timeframe, Ichimoku cloud breakout with 1-week EMA50 filter for trend alignment captures institutional moves in both bull and bear markets. The cloud acts as dynamic support/resistance, reducing whipsaws. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Ichimoku Cloud on primary timeframe (6h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Current cloud boundaries (Senkou Span A/B shifted back 26 periods to align with price)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Long: price breaks above cloud + above weekly EMA50
            if price_high > cloud_top and price_close > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + below weekly EMA50
            elif price_low < cloud_bottom and price_close < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters cloud or trend weakens
            if position == 1:
                if price_low < cloud_bottom or price_close < ema_50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_high > cloud_top or price_close > ema_50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_v1"
timeframe = "6h"
leverage = 1.0