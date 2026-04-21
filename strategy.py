#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: 6h Ichimoku cloud twist (Tenkan/Kijun cross) filtered by 1d EMA50 trend and volume spike (1.8x).
In bull markets (price > EMA50_1d): long on bullish TK cross above cloud, short on bearish TK cross below cloud.
In bear markets (price < EMA50_1d): only short on bearish TK cross below cloud, long on bullish TK cross above cloud only if price > cloud (counter-trend long only in strong bounce).
Volume confirmation reduces false signals. Discrete position sizing (0.25) limits fee drawdown.
Ichimoku works well in crypto due to its adaptive cloud structure. Timeframe: 6h, uses 1d HTF for trend filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for EMA50 trend ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Ichimoku components (calculate on 6h data) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for Ichimoku (52+26)
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) 
            or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        tenkan_now = tenkan[i]
        kijun_now = kijun[i]
        senkou_a_now = senkou_a[i]
        senkou_b_now = senkou_b[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Ichimoku cloud: top and bottom of cloud
        cloud_top = max(senkou_a_now, senkou_b_now)
        cloud_bottom = min(senkou_a_now, senkou_b_now)
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        # TK cross signals
        tk_cross_bullish = tenkan_now > kijun_now and tenkan[i-1] <= kijun[i-1]
        tk_cross_bearish = tenkan_now < kijun_now and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Determine market regime based on 1d EMA50
            if price > ema_trend:  # Bull regime
                # Long: bullish TK cross above cloud
                long_condition = tk_cross_bullish and price > cloud_top and volume_confirmed
                # Short: bearish TK cross below cloud
                short_condition = tk_cross_bearish and price < cloud_bottom and volume_confirmed
            else:  # Bear regime
                # Short: bearish TK cross below cloud (trend continuation)
                short_condition = tk_cross_bearish and price < cloud_bottom and volume_confirmed
                # Long: bullish TK cross above cloud ONLY if price > cloud (strong bounce)
                long_condition = tk_cross_bullish and price > cloud_top and price > ema_trend * 1.02 and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: bearish TK cross OR price drops below cloud bottom
            if tk_cross_bearish or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud top
            if tk_cross_bullish or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0