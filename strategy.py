#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Volume + 1d Trend Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance with leading signals.
Combining with volume confirms institutional participation. Using 1d trend (price > EMA50)
as filter ensures we trade in direction of higher timeframe trend.
Works in bull markets (buy on bullish TK cross above cloud) and bear markets (sell on bearish TK cross below cloud).
Target: 60-120 trades over 4 years (15-30/year) to balance opportunity and fee cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    # Not used in signals as it requires future data
    
    # Cloud top and bottom (for current price)
    # Senkou A and B are plotted 26 periods ahead, so we need to shift them back 26 to compare with current price
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN (no data)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top is max of Senkou A and B, cloud bottom is min
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (need 52 periods for Senkou B)
    start = max(52, 26)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below cloud OR TK cross turns bearish
            if close[i] < cloud_bottom[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above cloud OR TK cross turns bullish
            if close[i] > cloud_top[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Volume filter (20-period average)
            if i >= 20:
                vol_ma = np.mean(volume[i-20:i])
                volume_filter = volume[i] > vol_ma * 1.5
            else:
                volume_filter = False
            
            # Trend filter from 1d: price above/below EMA50
            price_above_ema = close[i] > ema_1d_aligned[i]
            price_below_ema = close[i] < ema_1d_aligned[i]
            
            # TK cross signals
            tk_bullish = tenkan[i] > kijun[i]
            tk_bearish = tenkan[i] < kijun[i]
            
            # Price relative to cloud
            price_above_cloud = close[i] > cloud_top[i]
            price_below_cloud = close[i] < cloud_bottom[i]
            
            # Look for entries: TK cross + volume + trend filter + price relative to cloud
            # Long: bullish TK cross above cloud in uptrend
            if tk_bullish and price_above_cloud and price_above_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross below cloud in downtrend
            elif tk_bearish and price_below_cloud and price_below_ema and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals