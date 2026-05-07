#!/usr/bin/env python3
"""
6h_ADX_Slope_Signal
Hypothesis: Use the slope of ADX(14) as a proxy for trend acceleration/deceleration. Enter long when ADX slope turns positive (trend strengthening) with price above 200 EMA for trend filter. Enter short when ADX slope turns negative with price below 200 EMA. Uses 1d timeframe for 200 EMA filter to avoid whipsaw. Designed for low-frequency, high-conviction entries.
"""
name = "6h_ADX_Slope_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate ADX(14)
    period = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_sum = wilders_smooth(tr, period)
    plus_dm_sum = wilders_smooth(plus_dm, period)
    minus_dm_sum = wilders_smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    
    # ADX slope: difference between current and previous ADX
    adx_slope = np.diff(adx, prepend=adx[0])
    
    # 200 EMA on 1d for trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 210  # Need sufficient warmup for ADX and EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(adx_slope[i]) or np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX slope turns positive (trend strengthening) + price above 200 EMA
            if (adx_slope[i] > 0 and adx_slope[i-1] <= 0 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX slope turns negative (trend weakening) + price below 200 EMA
            elif (adx_slope[i] < 0 and adx_slope[i-1] >= 0 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: ADX slope reverses direction
            if position == 1:
                if adx_slope[i] < 0 and adx_slope[i-1] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if adx_slope[i] > 0 and adx_slope[i-1] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals