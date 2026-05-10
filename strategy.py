#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Use 12h EMA50 trend filter and 1d volume spike with Camarilla R1/S1 breakout on 4h.
# Long when 12h trend up, volume > 2x average, and price breaks above Camarilla R1.
# Short when 12h trend down, volume > 2x average, and price breaks below Camarilla S1.
# Designed for low trade frequency (20-50/year) to avoid fee drag, works in bull/bear via trend filter.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA50 trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # 1d volume spike filter: current volume > 2 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (2 * vol_ma)
    
    # Align 1d volume spike to 4h
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # 4h Camarilla levels (based on previous day)
    # Calculate from previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 for each day
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h uptrend + volume spike + break above R1
            if (trend_12h_up_aligned[i] > 0.5 and 
                vol_spike_aligned[i] > 0.5 and
                close[i] > r1_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend + volume spike + break below S1
            elif (trend_12h_down_aligned[i] > 0.5 and 
                  vol_spike_aligned[i] > 0.5 and
                  close[i] < s1_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: 12h trend fails or volume drops
            if (trend_12h_up_aligned[i] < 0.5 or 
                vol_spike_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: 12h trend fails or volume drops
            if (trend_12h_down_aligned[i] < 0.5 or 
                vol_spike_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals