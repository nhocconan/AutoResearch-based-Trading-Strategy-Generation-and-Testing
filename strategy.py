#!/usr/bin/env python3
# 6h_Pivots_Reversal_1dTrend_VolumeFilter
# Hypothesis: Daily pivot points (classical) provide key support/resistance levels.
# Long when price bounces from S1/S2 with bullish daily trend and volume spike.
# Short when price rejects at R1/R2 with bearish daily trend and volume spike.
# Uses 60-minute close to avoid intrabar noise. Targets 15-25 trades/year.

name = "6h_Pivots_Reversal_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily pivots from previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Classical pivot point: (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Support and resistance levels
    s1 = 2 * pivot - prev_high
    s2 = pivot - (prev_high - prev_low)
    r1 = 2 * pivot - prev_low
    r2 = pivot + (prev_high - prev_low)
    
    # Align daily levels to 6h timeframe (available after daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Daily EMA for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(daily_ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price near support with bullish daily trend and volume spike
            near_support = (close[i] <= s1_aligned[i] * 1.005) or (close[i] <= s2_aligned[i] * 1.005)
            bullish_trend = close[i] > daily_ema_aligned[i]
            volume_spike = volume[i] > vol_ma[i] * 2.0
            
            if near_support and bullish_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near resistance with bearish daily trend and volume spike
            elif (close[i] >= r1_aligned[i] * 0.995) or (close[i] >= r2_aligned[i] * 0.995):
                if close[i] < daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or daily trend turns bearish
            if close[i] >= pivot_aligned[i] or close[i] < daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or daily trend turns bullish
            if close[i] <= pivot_aligned[i] or close[i] > daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals