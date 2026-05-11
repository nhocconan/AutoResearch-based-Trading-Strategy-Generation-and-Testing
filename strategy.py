#!/usr/bin/env python3
"""
4h_12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses daily EMA34 for trend direction, Camarilla R3/S3 levels on 12h chart for entry/exit,
with volume confirmation. Designed to capture breakouts from key pivot levels in both bull and bear
markets by following higher-timeframe trend. Targets low trade frequency (20-30/year) via strict
Camarilla level breaks and volume filters.
"""

name = "4h_12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily EMA34 for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = np.where(df_1d['close'].values > ema34_1d, 1, -1)
    trend_1d_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # --- 12h Camarilla Levels ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    r1_12h, r2_12h, r3_12h, r4_12h, s1_12h, s2_12h, s3_12h, s4_12h = calculate_camarilla(
        df_12h['high'].values, df_12h['low'].values, df_12h['close'].values
    )
    
    # Align Camarilla levels to 4h timeframe
    r3_12h_4h = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_4h = align_htf_to_ltf(prices, df_12h, s3_12h)
    
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
        if (np.isnan(trend_1d_4h[i]) or np.isnan(r3_12h_4h[i]) or 
            np.isnan(s3_12h_4h[i]) or np.isnan(vol_ratio[i])):
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
            # Long: uptrend + price breaks above R3 + volume
            if (trend_1d_4h[i] == 1 and 
                close[i] > r3_12h_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price breaks below S3 + volume
            elif (trend_1d_4h[i] == -1 and 
                  close[i] < s3_12h_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or price returns to opposite S3/R3
            if position == 1:
                # Exit long: trend turns down OR price closes below S3
                if trend_1d_4h[i] == -1 or close[i] < s3_12h_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price closes above R3
                if trend_1d_4h[i] == 1 or close[i] > r3_12h_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals