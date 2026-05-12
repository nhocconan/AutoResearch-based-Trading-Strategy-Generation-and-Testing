#!/usr/bin/env python3
# 1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_1DVOLUME
# Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter (EMA50) and 1d volume spike.
# Uses 4h for trend direction, 1d for volume confirmation, 1h for entry timing.
# Works in bull/bear: trend filter avoids counter-trend, volume spike confirms institutional interest.
# Target: 15-35 trades/year on 1h timeframe.

name = "1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_1DVOLUME"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # EMA50 for 4h trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (1.5 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Previous day's Camarilla levels (using prior day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_width = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 4h uptrend + volume spike + price breaks above R1
            if (close[i] > ema50_4h_aligned[i] and 
                vol_spike_1d_aligned[i] > 0.5 and  # bool treated as 0/1
                close[i] > r1_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + volume spike + price breaks below S1
            elif (close[i] < ema50_4h_aligned[i] and 
                  vol_spike_1d_aligned[i] > 0.5 and
                  close[i] < s1_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price below S1 (reversion to mean)
            if (close[i] <= ema50_4h_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Trend reversal or price above R1 (reversion to mean)
            if (close[i] >= ema50_4h_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals