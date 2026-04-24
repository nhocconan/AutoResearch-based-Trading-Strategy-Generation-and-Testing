#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for trend direction (price above/below 1d EMA50).
- Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (26/52-period).
- Entry: Long when price breaks above cloud AND Tenkan > Kijun (bullish TK cross) AND 1d EMA50 up.
         Short when price breaks below cloud AND Tenkan < Kijun (bearish TK cross) AND 1d EMA50 down.
         Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Exit: Opposite TK cross or price re-enters cloud.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2.0
    
    # The actual cloud boundaries at current price (shifted back by 26 to align)
    # Senkou Span A and B are plotted 26 periods ahead, so to get current cloud we look back 26
    shift = period_kijun  # 26 periods
    senkou_a_shifted = np.roll(senkou_a, shift)
    senkou_b_shifted = np.roll(senkou_b, shift)
    # Fill first 'shift' values with NaN (not yet available)
    senkou_a_shifted[:shift] = np.nan
    senkou_b_shifted[:shift] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need: 52 for Senkou B, 26 for Kijun, 9 for Tenkan, plus 26 shift for cloud
    start_idx = max(60, 52 + 26, 50)  # 1d EMA50 warmup, Ichimoku warmup, cloud shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema50_trend = ema50_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish: price above cloud, Tenkan > Kijun (bullish TK cross), 1d EMA50 rising
                if (curr_close > cloud_top[i] and 
                    tenkan[i] > kijun[i] and 
                    ema50_trend > ema50_1d_aligned[max(i-1, start_idx)]):  # 1d EMA50 rising
                    signals[i] = 0.25
                    position = 1
                # Bearish: price below cloud, Tenkan < Kijun (bearish TK cross), 1d EMA50 falling
                elif (curr_close < cloud_bottom[i] and 
                      tenkan[i] < kijun[i] and 
                      ema50_trend < ema50_1d_aligned[max(i-1, start_idx)]):  # 1d EMA50 falling
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters cloud OR bearish TK cross
            if (curr_close < cloud_top[i] and curr_close > cloud_bottom[i]) or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters cloud OR bullish TK cross
            if (curr_close < cloud_top[i] and curr_close > cloud_bottom[i]) or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0