#!/usr/bin/env python3
# 6h_Supertrend_Ichimoku_Breakout
# Hypothesis: Combines Supertrend trend detection with Ichimoku cloud breakouts to capture strong momentum moves.
# Uses 1d Supertrend for trend direction and 1d Ichimoku for entry/exit signals. Designed for low trade frequency
# (15-25/year) to minimize fee drift while capturing major trend moves in both bull and bear markets.

name = "6h_Supertrend_Ichimoku_Breakout"
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
    
    # Get 1d data for Supertrend and Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend on 1d
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full_like(tr, np.nan)
    for i in range(atr_period, len(tr)):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)
    
    for i in range(atr_period, len(close_1d)):
        if i == atr_period:
            supertrend[i] = lowerband[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upperband[i-1]:
                if close_1d[i] <= upperband[i]:
                    supertrend[i] = upperband[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
            else:
                if close_1d[i] >= lowerband[i]:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upperband[i]
                    direction[i] = -1
    
    # Calculate Ichimoku on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full_like(close_1d, np.nan)
    for i in range(period_tenkan-1, len(close_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full_like(close_1d, np.nan)
    for i in range(period_kijun-1, len(close_1d)):
        kijun_sen[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    senkou_span_b = np.full_like(close_1d, np.nan)
    for i in range(period_senkou_b-1, len(close_1d)):
        senkou_span_b[i] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou_span = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)-period_kijun):
        chikou_span[i] = close_1d[i+period_kijun]
    
    # Align all indicators to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Volume confirmation (20-period average on 6h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 52) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or \
           np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or \
           np.isnan(chikou_span_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Cloud color: green when Senkou Span A > Senkou Span B, red when opposite
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        cloud_red = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        if position == 0:
            # Long: Supertrend uptrend, price above cloud, TK cross bullish, volume confirmation
            if (direction_aligned[i] == 1 and 
                close[i] > senkou_span_a_aligned[i] and 
                close[i] > senkou_span_b_aligned[i] and
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend, price below cloud, TK cross bearish, volume confirmation
            elif (direction_aligned[i] == -1 and 
                  close[i] < senkou_span_a_aligned[i] and 
                  close[i] < senkou_span_b_aligned[i] and
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend turns down OR price falls below cloud OR TK cross bearish
            if (direction_aligned[i] == -1 or 
                close[i] < senkou_span_a_aligned[i] or 
                close[i] < senkou_span_b_aligned[i] or
                tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend turns up OR price rises above cloud OR TK cross bullish
            if (direction_aligned[i] == 1 or 
                close[i] > senkou_span_a_aligned[i] or 
                close[i] > senkou_span_b_aligned[i] or
                tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals