#!/usr/bin/env python3
"""
1d_1w_Keltner_Channel_Breakout_TrendFilter
Hypothesis: Uses weekly SuperTrend for trend direction, with price breaking above/below
Keltner Channel on daily chart as entry signal. Requires volume confirmation. Designed to work in both
bull and bear markets by following higher-timeframe trend while using daily for precise entries and exits.
Targets low trade frequency (7-25/year) via weekly trend filter and daily Keltner breakout.
"""

name = "1d_1w_Keltner_Channel_Breakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First TR value (set to high-low)
    tr[0] = tr1[0]
    
    # Average True Range
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    
    # Initialize first values
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close)):
        # Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # SuperTrend
    supertrend = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = final_ub[i]
        elif supertrend[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
            supertrend[i] = final_ub[i]
        elif supertrend[i-1] == final_ub[i-1] and close[i] > final_ub[i]:
            supertrend[i] = final_lb[i]
        elif supertrend[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
            supertrend[i] = final_lb[i]
        elif supertrend[i-1] == final_lb[i-1] and close[i] < final_lb[i]:
            supertrend[i] = final_ub[i]
    
    # Trend direction: 1 for uptrend, -1 for downtrend
    trend = np.where(supertrend <= close, 1, -1)
    
    return supertrend, trend, atr

def calculate_keltner_channel(high, low, close, period=20, multiplier=2):
    """Calculate Keltner Channel"""
    # Typical Price
    tp = (high + low + close) / 3
    
    # EMA of Typical Price
    ema_tp = pd.Series(tp).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Average True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Upper and Lower Bands
    upper = ema_tp + multiplier * atr
    lower = ema_tp - multiplier * atr
    
    return upper, lower, ema_tp

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly SuperTrend for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    supertrend_1w, trend_1w, atr_1w = calculate_supertrend(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Align weekly SuperTrend to daily timeframe
    supertrend_1w_1d = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    trend_1w_1d = align_htf_to_ltf(prices, df_1w, trend_1w)
    atr_1w_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # --- Daily Keltner Channel ---
    kc_upper, kc_lower, kc_middle = calculate_keltner_channel(
        high, low, close, period=20, multiplier=2
    )
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_1w_1d[i]) or np.isnan(trend_1w_1d[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: uptrend + price breaks above Keltner upper + volume
            if (trend_1w_1d[i] == 1 and 
                close[i] > kc_upper[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price breaks below Keltner lower + volume
            elif (trend_1w_1d[i] == -1 and 
                  close[i] < kc_lower[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or price returns to middle
            if position == 1:
                # Exit long: trend turns down OR price closes below middle
                if trend_1w_1d[i] == -1 or close[i] < kc_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price closes above middle
                if trend_1w_1d[i] == 1 or close[i] > kc_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals