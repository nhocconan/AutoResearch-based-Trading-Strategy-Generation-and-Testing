#!/usr/bin/env python3
# 6h_ADX_Ichimoku_Cloud_Filter
# Hypothesis: Combine ADX trend strength with Ichimoku cloud for high-probability breakouts.
# In trending markets (ADX>25), price above/below cloud confirms direction.
# Cloud acts as dynamic support/resistance, reducing false breakouts.
# Works in bull/bear by following ADX trend direction. Target: 50-150 trades over 4 years.

name = "6h_ADX_Ichimoku_Cloud_Filter"
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

    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Calculate Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = np.array([np.max(high_1d[i-8:i+1]) if i>=8 else np.nan for i in range(len(high_1d))])
    period9_low = np.array([np.min(low_1d[i-8:i+1]) if i>=8 else np.nan for i in range(len(low_1d))])
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = np.array([np.max(high_1d[i-25:i+1]) if i>=25 else np.nan for i in range(len(high_1d))])
    period26_low = np.array([np.min(low_1d[i-25:i+1]) if i>=25 else np.nan for i in range(len(low_1d))])
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = np.array([np.max(high_1d[i-51:i+1]) if i>=51 else np.nan for i in range(len(high_1d))])
    period52_low = np.array([np.min(low_1d[i-51:i+1]) if i>=51 else np.nan for i in range(len(low_1d))])
    senkou_b = ((period52_high + period52_low) / 2)

    # The actual cloud boundaries (shifted back to align with current price)
    # Senkou Span A and B are plotted 26 periods ahead, so to get current cloud we use values from 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan

    # Align Ichimoku cloud to 6h
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)

    # Calculate ADX on 6h data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            
            # Fix: if both are positive, set the smaller to 0
            if plus_dm[i] > 0 and minus_dm[i] > 0:
                if plus_dm[i] > minus_dm[i]:
                    minus_dm[i] = 0
                else:
                    plus_dm[i] = 0
                    
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        adx = np.zeros(len(high))
        
        # Initial values
        atr[period] = np.sum(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i]
            minus_di[i] = 100 * minus_dm_sum / atr[i]
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
            
        # ADX is smoothed DX
        adx[2*period] = np.sum(dx[period+1:2*period+1]) / period
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx

    adx = calculate_adx(high, low, close, 14)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data is not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
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
            # LONG: price above cloud, ADX>25 (strong trend), volume spike
            if close[i] > cloud_top and adx[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud, ADX>25 (strong trend), volume spike
            elif close[i] < cloud_bottom and adx[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below cloud bottom or ADX weakens
            if close[i] < cloud_bottom or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above cloud top or ADX weakens
            if close[i] > cloud_top or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals