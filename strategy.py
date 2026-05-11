#!/usr/bin/env python3
"""
6h_Ichimoku_KumoBreakout_1dTrend_MultiTF_Filter
Hypothesis: Use Ichimoku cloud (Tenkan/Kijun/Senkou A/B) on 6h with trend filter from 1d EMA50 and volume confirmation.
Trades only when price breaks above/below cloud with TK cross in same direction, aligned with 1d trend.
Designed for 60-120 total trades over 4 years (15-30/year) to avoid fee drift.
Works in bull/bear via 1d trend filter: only long when above 1d EMA50, short when below.
"""

name = "6h_Ichimoku_KumoBreakout_1dTrend_MultiTF_Filter"
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
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d EMA50 Trend Filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Ichimoku on 6h (Tenkan, Kijun, Senkou A/B) ===
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Shift Senkou A/B by 26 periods (cloud is plotted 26 periods ahead)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Volume spike filter (20-period EMA)
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Signal parameters
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers Ichimoku calculations)
    start_idx = 80  # covers 52 + 26 shift
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above cloud + TK cross bullish + above 1d EMA50 + volume spike
            if (close[i] > cloud_top[i] and 
                tenkan[i] > kijun[i] and 
                close[i] > ema50_1d_6h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Price below cloud + TK cross bearish + below 1d EMA50 + volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  close[i] < ema50_1d_6h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Hold for minimum 6 bars to reduce whipsaw
            holding_bars += 1
            if holding_bars < 6:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: TK cross reverses OR price re-enters cloud
            if position == 1:
                if tenkan[i] < kijun[i] or close[i] < cloud_top[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if tenkan[i] > kijun[i] or close[i] > cloud_bottom[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals