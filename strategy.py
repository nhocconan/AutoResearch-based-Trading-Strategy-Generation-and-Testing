#!/usr/bin/env python3
"""
6h_adx_aroon_1d_trend_v1
Hypothesis: On 6h timeframe, use daily ADX for trend strength and Aroon for trend direction to capture strong trending moves while avoiding choppy markets. Enter long when ADX > 25 and Aroon Up > Aroon Down; enter short when ADX > 25 and Aroon Down > Aroon Up. Exit when ADX falls below 20 (trend weakness). This strategy targets strong trending moves with ADX filter to reduce false signals and trade frequency. Works in bull/bear via trend strength filter and direction from Aroon.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_aroon_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for ADX and Aroon
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    period = 14
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]),
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]),
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+
    tr_period = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    dm_plus_period = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum().values
    dm_minus_period = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum().values
    # Avoid division by zero
    tr_period_safe = np.where(tr_period == 0, 1e-10, tr_period)
    # DI+ and DI-
    di_plus = 100 * dm_plus_period / tr_period_safe
    di_minus = 100 * dm_minus_period / tr_period_safe
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # Calculate Aroon (14-period)
    # Aroon Up: ((period - periods since highest high) / period) * 100
    # Aroon Down: ((period - periods since lowest low) / period) * 100
    def calculate_aroon(arr, period):
        n_arr = len(arr)
        aroon_up = np.full(n_arr, np.nan)
        aroon_down = np.full(n_arr, np.nan)
        for i in range(period-1, n_arr):
            window_high = arr[i-period+1:i+1]
            window_low = arr[i-period+1:i+1]
            highest_high_idx = np.argmax(window_high)
            lowest_low_idx = np.argmin(window_low)
            periods_since_high = period - 1 - highest_high_idx
            periods_since_low = period - 1 - lowest_low_idx
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        return aroon_up, aroon_down
    
    aroon_up, aroon_down = calculate_aroon(high_1d, period)
    aroon_down = calculate_aroon(low_1d, period)[1]  # Recalculate for lows
    
    # Align indicators to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    aroon_up_6h = align_htf_to_ltf(prices, df_1d, aroon_up)
    aroon_down_6h = align_htf_to_ltf(prices, df_1d, aroon_down)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_6h[i]) or np.isnan(aroon_up_6h[i]) or np.isnan(aroon_down_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit if trend weakens (ADX < 20)
            if adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend weakens (ADX < 20)
            if adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: strong trend (ADX > 25) and bullish direction (Aroon Up > Aroon Down)
            if adx_6h[i] > 25 and aroon_up_6h[i] > aroon_down_6h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: strong trend (ADX > 25) and bearish direction (Aroon Down > Aroon Up)
            elif adx_6h[i] > 25 and aroon_down_6h[i] > aroon_up_6h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals