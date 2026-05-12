#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Rebound"
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
    
    # === 1d data ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Camarilla pivot levels ===
    rango = high_1d - low_1d
    camarilla_r1 = close_1d + (rango * 1.1 / 12)
    camarilla_s1 = close_1d - (rango * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d EMA34 trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Volume spike filter ===
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === 4h RSI for pullback confirmation ===
    close_series = pd.Series(close)
    rsi_gain = close_series.diff().clip(lower=0)
    rsi_loss = -close_series.diff().clip(upper=0)
    avg_gain = rsi_gain.rolling(window=14, min_periods=14).mean()
    avg_loss = rsi_loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R1 + above daily EMA34 + volume spike + RSI pullback (30-50)
            if (close[i] > camarilla_r1_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5 and
                30 <= rsi[i] <= 50):
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 + below daily EMA34 + volume spike + RSI pullback (50-70)
            elif (close[i] < camarilla_s1_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5 and
                  50 <= rsi[i] <= 70):
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