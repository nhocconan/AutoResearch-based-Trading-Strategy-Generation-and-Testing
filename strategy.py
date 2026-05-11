#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_4hTrend_1dVolume_Combined
Hypothesis: Use 4h EMA50 for trend direction, 1d volume spike for conviction, and 1h Camarilla R1/S1 breakouts for entry. This combines multiple timeframe confirmation to reduce false signals. Works in bull markets (buy R1 breaks in uptrend) and bear markets (sell S1 breaks in downtrend). Volume spike confirms institutional interest. Target: 15-30 trades per year on 1h timeframe with 4h/1d filters.
"""

name = "1h_Camarilla_R1_S1_4hTrend_1dVolume_Combined"
timeframe = "1h"
leverage = 1.0

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
    
    # === 4H Data for Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1D Data for Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 1d volume spike: current volume > 2x 20-day average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # === 1H Data for Camarilla Levels ===
    # Previous hour's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First hour uses same hour's high
    prev_low[0] = low[0]    # First hour uses same hour's low
    prev_close[0] = close[0] # First hour uses same hour's close
    
    # Camarilla levels: R1/S1 = C ± (H-L) * 1.1/12
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(r1[i]) or 
            np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 4h uptrend (price > EMA50) AND 1d volume spike
            if close[i] > r1[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_1d_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND 4h downtrend (price < EMA50) AND 1d volume spike
            elif close[i] < s1[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_1d_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA50 OR reverses below R1
            if close[i] < ema_50_4h_aligned[i] or close[i] < r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA50 OR reverses above S1
            if close[i] > ema_50_4h_aligned[i] or close[i] > s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals