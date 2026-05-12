#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === DAILY CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla (avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    range_1d = prev_high_1d - prev_low_1d
    camarilla_base = prev_close_1d
    
    # Resistance and Support levels
    r1 = camarilla_base + range_1d * 1.1 / 6
    r2 = camarilla_base + range_1d * 1.1 / 4
    s1 = camarilla_base - range_1d * 1.1 / 6
    s2 = camarilla_base - range_1d * 1.1 / 4
    
    # Align to daily timeframe (same as prices for 1d timeframe)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_1d[i]) or np.isnan(r2_1d[i]) or 
            np.isnan(s1_1d[i]) or np.isnan(s2_1d[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R2 with volume spike + price above weekly EMA50 (uptrend)
            if (close[i] > r2_1d[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S2 with volume spike + price below weekly EMA50 (downtrend)
            elif (close[i] < s2_1d[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below R1 (false breakout) OR below weekly EMA50
            if close[i] < r1_1d[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S1 (false breakout) OR above weekly EMA50
            if close[i] > s1_1d[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals