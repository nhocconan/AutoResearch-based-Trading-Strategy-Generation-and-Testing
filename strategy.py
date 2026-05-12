#!/usr/bin/env python3
# 6h_1W_1D_Ichimoku_CloudBreakout_Trend
# Hypothesis: Ichimoku Cloud breakout on 6h timeframe filtered by 1d trend (price > EMA50) and 1w momentum (price > EMA200).
# Long when price breaks above Kumo (cloud) with bullish TK cross, short when breaks below with bearish TK cross.
# Works in bull markets via trend-following breakouts and in bear via mean-reversion from cloud edges.
# Weekly EMA200 avoids false signals in weak trends; daily EMA50 ensures alignment with intermediate trend.

name = "6h_1W_1D_Ichimoku_CloudBreakout_Trend"
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
    
    # 6h Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For breakout detection, we use current cloud (shifted 26 periods ago)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly momentum filter: EMA200
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup for Ichimoku and EMAs
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above cloud + bullish TK cross + price > 1d EMA50 + price > 1w EMA200
            if (close[i] > cloud_top[i] and 
                tk_cross_up[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + bearish TK cross + price < 1d EMA50 + price < 1w EMA200
            elif (close[i] < cloud_bottom[i] and 
                  tk_cross_down[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud OR bearish TK cross OR price < 1d EMA50
            if (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]) or \
               tk_cross_down[i] or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud OR bullish TK cross OR price > 1d EMA50
            if (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]) or \
               tk_cross_up[i] or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals