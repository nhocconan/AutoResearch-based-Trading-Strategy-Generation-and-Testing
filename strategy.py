# 2025-06-22 | 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA trend filter and volume confirmation.
# Long when price breaks above R1 with volume > 1.5x average and price > 1w EMA50.
# Short when price breaks below S1 with volume > 1.5x average and price < 1w EMA50.
# Designed for 1d timeframe to capture swing moves with low trade frequency (target: 15-30 trades/year).
# Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear via trend filter.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50
    
    # Align 1w EMA to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d OHLC for Camarilla (use daily data directly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla R1 and S1 for each day using prior day's HLC
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        camarilla_R1[i] = prev_close + range_val * 1.1 / 12
        camarilla_S1[i] = prev_close - range_val * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (already aligned via index)
    camarilla_R1_aligned = camarilla_R1
    camarilla_S1_aligned = camarilla_S1
    
    # Volume filter: 1d volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Ensure volume MA and at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 AND volume confirmation AND price > 1w EMA
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 1.5 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 AND volume confirmation AND price < 1w EMA
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 1.5 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (reversal)
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal)
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals