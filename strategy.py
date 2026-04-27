#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 1d for trend direction, volume confirmation, and choppiness index regime filter to avoid whipsaws in ranging markets.
Long when KAMA slope > 0 AND volume > 1.5 * 20-period average AND choppiness < 61.8 (trending regime).
Short when KAMA slope < 0 AND volume > 1.5 * 20-period average AND choppiness < 61.8 (trending regime).
Exit when trend reverses or choppiness > 61.8 (range regime).
Designed for 1d timeframe to achieve 30-100 total trades over 4 years with low fee drag.
Uses 1w HTF for trend confirmation (only trade in direction of 1w KAMA).
Works in both bull and bear markets by following adaptive trend while filtering sideways action.
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
    
    # Get 1d data for KAMA and chop calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1d KAMA calculation
    close_1d_series = pd.Series(df_1d['close'].values)
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series - close_1d_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d_series)
    kama[0] = close_1d_series.iloc[0]
    for i in range(1, len(close_1d_series)):
        kama[i] = kama[i-1] + sc[i] * (close_1d_series.iloc[i] - kama[i-1])
    kama = kama
    
    # 1d KAMA slope (5-period difference)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # 1d Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR1) / (n * log2(n+1))) / log2(n)
    tr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    n = 14
    chop = 100 * np.log10(sum_atr1 / (n * np.log2(n + 1))) / np.log2(n)
    
    # 1d Volume confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # 1w KAMA for HTF trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    change_w = abs(close_1w_series - close_1w_series.shift(10))
    volatility_w = abs(close_1w_series - close_1w_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er_w = change_w / volatility_w.replace(0, np.nan)
    er_w = er_w.fillna(0).values
    sc_w = (er_w * (fast_sc - slow_sc) + slow_sc) ** 2
    kama_w = np.zeros_like(close_1w_series)
    kama_w[0] = close_1w_series.iloc[0]
    for i in range(1, len(close_1w_series)):
        kama_w[i] = kama_w[i-1] + sc_w[i] * (close_1w_series.iloc[i] - kama_w[i-1])
    kama_w_slope = np.diff(kama_w, prepend=kama_w[0])
    
    # Align all 1d indicators to lower timeframe (prices index)
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    kama_w_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_w_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), chop (14), volume avg (20), 1w KAMA (10)
    start_idx = max(10, 14, 20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(kama_w_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_slope_val = kama_slope_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm_aligned[i] > 0.5  # convert back to boolean
        kama_w_slope_val = kama_w_slope_aligned[i]
        
        if position == 0:
            # Look for entry: KAMA slope aligned with 1w trend AND volume AND trending regime (chop < 61.8)
            long_condition = (kama_slope_val > 0) and (kama_w_slope_val > 0) and vol_conf and (chop_val < 61.8)
            short_condition = (kama_slope_val < 0) and (kama_w_slope_val < 0) and vol_conf and (chop_val < 61.8)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when KAMA slope turns negative OR chop > 61.8 (range) OR 1w trend breaks
            exit_condition = (kama_slope_val <= 0) or (chop_val > 61.8) or (kama_w_slope_val <= 0)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when KAMA slope turns positive OR chop > 61.8 (range) OR 1w trend breaks
            exit_condition = (kama_slope_val >= 0) or (chop_val > 61.8) or (kama_w_slope_val >= 0)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0