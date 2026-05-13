#!/usr/bin/env python3
"""
12h_Ichimoku_Bullish_Bearish_Balance
Hypothesis: Ichimoku Cloud components provide multi-timeframe equilibrium signals. Price above/below cloud with TK cross and Kumo twist gives high-probability trend continuation. Uses 1w trend filter and 1d volume confirmation to reduce false signals. Targets 15-25 trades/year on 12h timeframe with 0.25 position size for controlled risk.
"""

name = "12h_Ichimoku_Bullish_Bearish_Balance"
timeframe = "12h"
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
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    # Get 1d data for volume confirmation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # For signal generation, we compare current price with Chikou (which is past close)
    chikou = close  # We'll use current price vs price 26 periods ago
    
    # 1w trend filter: EMA(50) on weekly close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume confirmation: current volume > 2.0x 24-period average (2 days)
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_filter = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Current Ichimoku values
        tenkan_i = tenkan[i]
        kijun_i = kijun[i]
        # Senkou spans need to be offset for cloud calculation
        senkou_a_i = senkou_a[i] if i < len(senkou_a) else senkou_a[-1]
        senkou_b_i = senkou_b[i] if i < len(senkou_b) else senkou_b[-1]
        # Cloud top and bottom
        cloud_top = max(senkou_a_i, senkou_b_i)
        cloud_bottom = min(senkou_a_i, senkou_b_i)
        # Chikou comparison: current price vs price 26 periods ago
        chikow_value = close[i - 26] if i >= 26 else close[0]
        
        if position == 0:
            # LONG: Price above cloud, TK cross up, Chikou above price 26 periods ago, 1w uptrend, volume confirmation
            if (close[i] > cloud_top and 
                tenkan_i > kijun_i and 
                tenkan[i-1] <= kijun[i-1] and  # TK cross just happened
                close[i] > chikow_value and
                close[i] > ema50_1w_aligned[i] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud, TK cross down, Chikou below price 26 periods ago, 1w downtrend, volume confirmation
            elif (close[i] < cloud_bottom and 
                  tenkan_i < kijun_i and 
                  tenkan[i-1] >= kijun[i-1] and  # TK cross just happened
                  close[i] < chikow_value and
                  close[i] < ema50_1w_aligned[i] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below cloud OR TK cross down OR 1w trend change
            if (close[i] < cloud_top or 
                tenkan_i < kijun_i or
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above cloud OR TK cross up OR 1w trend change
            if (close[i] > cloud_bottom or 
                tenkan_i > kijun_i or
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals