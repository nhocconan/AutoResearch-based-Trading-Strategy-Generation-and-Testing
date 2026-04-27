#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close (1d shift for completed day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels (R4 and S4) - more extreme levels
    range_hl = prev_high - prev_low
    R4 = prev_close + range_hl * 1.1 / 2
    S4 = prev_close - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.5 * 24-period average (48h lookback)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 24) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(R4_4h[i]) or np.isnan(S4_4h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R4 + 1-day uptrend + volume spike
            if (close[i] > R4_4h[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S4 + 1-day downtrend + volume spike
            elif (close[i] < S4_4h[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S4 (reversal) or trend changes
            if (close[i] < S4_4h[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R4 (reversal) or trend changes
            if (close[i] > R4_4h[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0