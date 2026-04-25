#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_ADX_Volume
Hypothesis: On 6h timeframe, use 1d Ichimoku cloud (Senkou Span A/B) as trend filter and 1d ADX > 25 for trend strength. Enter long when 6h price breaks above cloud + Tenkan-Kijun cross bullish + volume spike. Enter short when price breaks below cloud + Tenkan-Kijun cross bearish + volume spike. Uses discrete sizing 0.25 to limit fee drag. Target 12-30 trades/year on 6h timeframe.
"""

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
    
    # Get 1d data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h (completed 1d bar only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate ADX on 1d for trend strength
    # +DM, -DM, TR
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - high_1d[:-1])),
        np.abs(low_1d[1:] - low_1d[:-1])
    )
    # Prepend first value as 0 for alignment
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Wilder's smoothing (ATR-like)
    def WilderSmooth(data, period):
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    period_adx = 14
    atr = WilderSmooth(tr, period_adx)
    plus_di = 100 * WilderSmooth(plus_dm, period_adx) / (atr + 1e-10)
    minus_di = 100 * WilderSmooth(minus_dm, period_adx) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = WilderSmooth(dx, period_adx)
    
    # Align ADX to 6h (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52), ADX (14*2+), volume MA (20)
    start_idx = max(52, 28, 20)  # 52 for Ichimoku SS B, 28 for ADX (14*2), 20 for volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Tenkan-Kijun cross
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: price above cloud + TK cross bullish + ADX > 25 + volume spike
            long_setup = (close[i] > cloud_top) and \
                         tk_cross_bullish and \
                         (adx_aligned[i] > 25) and \
                         volume_spike[i]
            # Short: price below cloud + TK cross bearish + ADX > 25 + volume spike
            short_setup = (close[i] < cloud_bottom) and \
                          tk_cross_bearish and \
                          (adx_aligned[i] > 25) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below cloud OR TK cross bearish
            if (close[i] < cloud_bottom) or \
               (tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above cloud OR TK cross bullish
            if (close[i] > cloud_top) or \
               (tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_ADX_Volume"
timeframe = "6h"
leverage = 1.0