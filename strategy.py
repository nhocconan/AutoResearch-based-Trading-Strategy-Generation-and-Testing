#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v1
Hypothesis: Trade Ichimoku cloud breakouts on 6h with 1d trend filter. Uses 1d EMA50 to determine trend direction (long only above, short only below). Enters when price breaks above/below cloud with TK cross confirmation and volume > 1.5x average. Avoids whipsaw by requiring alignment with higher timeframe trend. Designed for 6-12 trades/year per symbol, targeting 50-120 total over 4 years.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 0.0377) + (ema_50[i-1] * 0.9623)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    highest_tenkan = np.full_like(high, np.nan)
    lowest_tenkan = np.full_like(low, np.nan)
    for i in range(period_tenkan-1, len(high)):
        highest_tenkan[i] = np.max(high[i - period_tenkan + 1:i + 1])
        lowest_tenkan[i] = np.min(low[i - period_tenkan + 1:i + 1])
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    highest_kijun = np.full_like(high, np.nan)
    lowest_kijun = np.full_like(low, np.nan)
    for i in range(period_kijun-1, len(high)):
        highest_kijun[i] = np.max(high[i - period_kijun + 1:i + 1])
        lowest_kijun[i] = np.min(low[i - period_kijun + 1:i + 1])
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = np.full_like(high, np.nan)
    lowest_senkou_b = np.full_like(low, np.nan)
    for i in range(period_senkou_b-1, len(high)):
        highest_senkou_b[i] = np.max(high[i - period_senkou_b + 1:i + 1])
        lowest_senkou_b[i] = np.min(low[i - period_senkou_b + 1:i + 1])
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Align Ichimoku components (no future shift needed as components use historical data)
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    # TK Cross: Tenkan crosses Kijun
    tk_cross = np.zeros(n, dtype=int)  # 1: bullish cross, -1: bearish cross
    for i in range(1, n):
        if not (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
                np.isnan(tenkan_aligned[i-1]) or np.isnan(kijun_aligned[i-1])):
            if tenkan_aligned[i-1] <= kijun_aligned[i-1] and tenkan_aligned[i] > kijun_aligned[i]:
                tk_cross[i] = 1  # bullish TK cross
            elif tenkan_aligned[i-1] >= kijun_aligned[i-1] and tenkan_aligned[i] < kijun_aligned[i]:
                tk_cross[i] = -1  # bearish TK cross
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i - vol_period:i])
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, vol_period)  # Senkou B needs 52 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Price above cloud + bullish TK cross + above 1d EMA50 + volume
            if (close[i] > upper_cloud and tk_cross[i] == 1 and 
                close[i] > ema_50_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + bearish TK cross + below 1d EMA50 + volume
            elif (close[i] < lower_cloud and tk_cross[i] == -1 and 
                  close[i] < ema_50_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below cloud or bearish TK cross or below 1d EMA50
            if (close[i] < lower_cloud or tk_cross[i] == -1 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above cloud or bullish TK cross or above 1d EMA50
            if (close[i] > upper_cloud or tk_cross[i] == 1 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0