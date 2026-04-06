#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud combined with 1-day RSI and 1-week volume confirmation.
# Ichimoku provides trend direction (price above/below cloud), momentum (TK cross), and support/resistance.
# RSI on daily timeframe filters for overbought/oversold conditions to avoid false breakouts.
# Volume confirmation on weekly timeframe ensures institutional participation.
# Designed for 6h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in both bull and bear markets: cloud acts as dynamic support/resistance,
# TK cross captures momentum shifts, RSI avoids chasing extremes.

name = "6h_ichimoku1d_rsi1w_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day Ichimoku Cloud (9, 26, 52)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full(len(close_1d), np.nan)
    for i in range(8, len(close_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full(len(close_1d), np.nan)
    for i in range(25, len(close_1d)):
        kijun_sen[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full(len(close_1d), np.nan)
    for i in range(25, len(close_1d)):
        idx = i + 26
        if idx < len(close_1d):
            senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = np.full(len(close_1d), np.nan)
    for i in range(51, len(close_1d)):
        idx = i + 26
        if idx < len(close_1d):
            senkou_span_b[idx] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.full(len(close_1d), np.nan)
    for i in range(26, len(close_1d)):
        chikou_span[i-26] = close_1d[i]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # 1-day RSI(14) for overbought/oversold filtering
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
        else:
            rsi[i] = 100
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of all indicators)
    start = max(52 + 26, 14, 4)  # Ichimoku needs 52+26, RSI needs 14, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below cloud or stoploss
            if (close[i] < cloud_bottom or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above cloud or stoploss
            if (close[i] > cloud_top or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: price above cloud, TK cross bullish, RSI not overbought
                if (close[i] > cloud_top and 
                    tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                    rsi_aligned[i] < 70):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price below cloud, TK cross bearish, RSI not oversold
                elif (close[i] < cloud_bottom and 
                      tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                      rsi_aligned[i] > 30):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals