#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily timeframe for breakout signals on 4h chart.
Confirms trend using daily EMA34 and requires volume spike for entry. Designed to work in both bull and bear
markets by combining price level breakouts with trend and volume filters. Targets low trade frequency (19-50/year)
via strict entry conditions.
"""

name = "4h_1d_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    pivot = (high + low + close) / 3
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    s2 = close - range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s4 = close - range_val * 1.1 / 2
    return r1, s1, r2, s2, r3, s3, r4, s4, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivots (R1/S1) for Breakout Signals ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    r1, s1, _, _, _, _, _, _, pivot = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # --- Daily EMA34 for Trend Filter ---
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(ema_34_4h[i]) or
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
            # Long: price breaks above R1 + above EMA34 + volume spike
            if (close[i] > r1_4h[i] and 
                close[i] > ema_34_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below EMA34 + volume spike
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema_34_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses back below/above pivot
            if position == 1:
                # Exit long: price crosses below pivot
                if close[i] < pivot_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above pivot
                if close[i] > pivot_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals