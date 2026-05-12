#/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d/1w trend filter and volume confirmation.
Ichimoku provides: 1) Tenkan/Kijun cross for momentum, 2) Kumo cloud for support/resistance, 
3) Senkou Span for future cloud thickness. 1d/1w filters ensure alignment with higher timeframe trend,
4) Volume spike confirms breakout strength. Works in bull/bear markets: 
- Bull: TK cross above cloud + price above cloud + volume spike = long
- Bear: TK cross below cloud + price below cloud + volume spike = short
Target: 20-30 trades/year (80-120 total over 4 years).
"""
name = "6h_Ichimoku_1dTrend_1wFilter_Volume"
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
    volume = prices['volume'].values
    
    # === 1d DATA FOR TREND FILTER (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1w DATA FOR TREND FILTER (EMA200) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === ICHIMOKU COMPONENTS (9, 26, 52 periods) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Using 1d as base for alignment
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === VOLUME CONFIRMATION (24-period) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 50, 200)  # Max of Ichimoku (52), 1d EMA (50), 1w EMA (200)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # LONG: TK cross bullish, price above cloud, aligned with 1d/1w trend, volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and  # TK cross bullish
                close[i] > cloud_top and                  # Price above cloud
                close[i] > ema50_1d_aligned[i] and        # Above 1d EMA50
                close[i] > ema200_1w_aligned[i] and       # Above 1w EMA200
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross bearish, price below cloud, aligned with 1d/1w trend, volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and  # TK cross bearish
                  close[i] < cloud_bottom and               # Price below cloud
                  close[i] < ema50_1d_aligned[i] and        # Below 1d EMA50
                  close[i] < ema200_1w_aligned[i] and       # Below 1w EMA200
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TK cross bearish OR price below cloud
            if (tenkan_aligned[i] < kijun_aligned[i]) or (close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish OR price above cloud
            if (tenkan_aligned[i] > kijun_aligned[i]) or (close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals