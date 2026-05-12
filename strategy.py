#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_RSI"
timeframe = "4h"
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
    
    # === 1d Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels
    rango = high_1d - low_1d
    camarilla_r1 = close_1d + (rango * 1.1 / 12)
    camarilla_s1 = close_1d - (rango * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d EMA34 trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === 1h RSI filter (momentum filter) ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(rsi_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R1 + above daily EMA34 + volume spike + RSI > 50
            if (close[i] > camarilla_r1_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5 and
                rsi_1h_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 + below daily EMA34 + volume spike + RSI < 50
            elif (close[i] < camarilla_s1_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5 and
                  rsi_1h_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below EMA34
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R1 or above EMA34
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals