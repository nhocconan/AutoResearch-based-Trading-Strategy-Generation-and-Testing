#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
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
    
    # Pre-calculate session hours (UTC) to filter trades
    hours = prices.index.hour
    
    # === 4h Camarilla pivot levels ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    rango_4h = high_4h - low_4h
    camarilla_r1_4h = close_4h + (rango_4h * 1.1 / 12)
    camarilla_s1_4h = close_4h - (rango_4h * 1.1 / 12)
    
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # === 4h Trend filter: EMA50 ===
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 4h Volume spike filter ===
    vol_4h = df_4h['volume'].values
    vol_avg_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = vol_4h > (2.0 * vol_avg_4h)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_4h_aligned[i]) or 
            np.isnan(camarilla_s1_4h_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_spike_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R1 + above 4h EMA50 + volume spike
            if (close[i] > camarilla_r1_4h_aligned[i] and
                close[i] > ema50_4h_aligned[i] and
                vol_spike_4h_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short: Close below S1 + below 4h EMA50 + volume spike
            elif (close[i] < camarilla_s1_4h_aligned[i] and
                  close[i] < ema50_4h_aligned[i] and
                  vol_spike_4h_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below EMA50
            if close[i] < camarilla_s1_4h_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Close above R1 or above EMA50
            if close[i] > camarilla_r1_4h_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals