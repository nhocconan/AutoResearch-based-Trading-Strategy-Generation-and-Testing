#!/usr/bin/env python3
"""
6h_Ichimoku_Trend_Cloud_v1
Hypothesis: On 6h timeframe, Ichimoku Cloud with 1d weekly trend filter provides robust trend following that works in both bull and bear markets. 
In bull regime (1d close > 1d EMA50), we take longs when price breaks above cloud and Tenkan > Kijun. 
In bear regime (1d close < 1d EMA50), we take shorts when price breaks below cloud and Tenkan < Kijun. 
Volume confirmation filters weak moves. Discrete sizing (0.25) controls fee churn. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Ichimoku Cloud (9, 26, 52) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud (Senkou Span shifted back 26 periods for alignment)
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 20  # max 5 days (20 * 6h = 120h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_current[i]) or np.isnan(senkou_b_current[i]) or
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        daily_ema = ema_50_1d_aligned[i]
        
        # Cloud boundaries
        upper_cloud = np.maximum(senkou_a_current[i], senkou_b_current[i])
        lower_cloud = np.minimum(senkou_a_current[i], senkou_b_current[i])
        
        # Trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price > upper cloud AND Tenkan > Kijun
                long_condition = (price > upper_cloud) and (tenkan[i] > kijun[i]) and volume_confirmed[i]
            else:  # bear regime
                # Bear regime: short when price < lower cloud AND Tenkan < Kijun
                short_condition = (price < lower_cloud) and (tenkan[i] < kijun[i]) and volume_confirmed[i]
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions
            if position == 1:
                # Exit long: price < lower cloud OR Tenkan < Kijun OR max hold reached
                if (price < lower_cloud) or (tenkan[i] < kijun[i]) or (bars_since_entry >= max_hold_bars):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price > upper cloud OR Tenkan > Kijun OR max hold reached
                if (price > upper_cloud) or (tenkan[i] > kijun[i]) or (bars_since_entry >= max_hold_bars):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Trend_Cloud_v1"
timeframe = "6h"
leverage = 1.0