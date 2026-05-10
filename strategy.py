#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Improved
# Hypothesis: Use 1d trend filter + Camarilla pivot levels from 1d for entry with volume confirmation on 4h.
# Long when 1d close > 1d EMA34 and price breaks above R1 with volume; short when 1d close < 1d EMA34 and price breaks below S1 with volume.
# Designed for low trade frequency (20-40/year) to avoid fee drift, works in bull/bear via trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Improved"
timeframe = "4h"
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
    
    # 1d data for trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align 1d indicators to 4h (wait for 1d bar to close)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: 1d uptrend + price breaks above R1 with volume
            if (trend_1d_up_aligned[i] > 0.5 and 
                close[i] > camarilla_r1_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend + price breaks below S1 with volume
            elif (trend_1d_down_aligned[i] > 0.5 and 
                  close[i] < camarilla_s1_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: 1d trend fails or price returns to pivot
            if (trend_1d_up_aligned[i] < 0.5 or 
                close[i] < camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: 1d trend fails or price returns to pivot
            if (trend_1d_down_aligned[i] < 0.5 or 
                close[i] > camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals