#!/usr/bin/env python3
name = "4h_PhaseAccumulation_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # === Phase Accumulation Indicator (LTF) ===
    # Hilbert Transform - Phase Accumulation
    # Using 4-bar difference to detect cycle phase
    diff4 = np.zeros(n)
    diff4[4:] = close[4:] - close[:-4]
    
    # Smooth the difference
    alpha = 0.07
    smoothed = np.zeros(n)
    smoothed[0] = diff4[0] if not np.isnan(diff4[0]) else 0
    for i in range(1, n):
        if np.isnan(diff4[i]):
            smoothed[i] = smoothed[i-1]
        else:
            smoothed[i] = alpha * diff4[i] + (1 - alpha) * smoothed[i-1]
    
    # === 12h Trend Filter (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 1d Volume Spike Filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.8 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(smoothed[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Phase accumulation rising (market becoming bullish) + above 12h EMA50 + volume spike
            if (smoothed[i] > smoothed[i-1] and  # Phase rising
                close[i] > ema50_12h_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Phase accumulation falling (market becoming bearish) + below 12h EMA50 + volume spike
            elif (smoothed[i] < smoothed[i-1] and  # Phase falling
                  close[i] < ema50_12h_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Phase accumulation turns down or closes below 12h EMA50
            if smoothed[i] < smoothed[i-1] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Phase accumulation turns up or closes above 12h EMA50
            if smoothed[i] > smoothed[i-1] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals