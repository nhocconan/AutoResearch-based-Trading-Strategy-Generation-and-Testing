#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot (1d) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R4 AND price > 1w EMA40 AND volume > 1.5x 24-period avg volume
# Short when price breaks below 1d Camarilla S4 AND price < 1w EMA40 AND volume > 1.5x 24-period avg volume
# Camarilla R4/S4 represent strong breakout levels from prior day's range
# 1w EMA40 filter ensures alignment with weekly trend, reducing counter-trend trades
# Volume confirmation adds conviction to breakouts
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w EMA40 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_40 = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40)
    
    # === 1d Camarilla pivot levels (using prior day's H/L/C) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    camarilla_r4 = close_1d + range_hl * 1.1 / 2  # R4 = Close + (Range * 1.1/2)
    camarilla_s4 = close_1d - range_hl * 1.1 / 2  # S4 = Close - (Range * 1.1/2)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Volume Confirmation (24-period average = 6 days, but we use 6 periods = 1.5 days) ===
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_40_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma_6[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_40_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        vol_confirm = volume[i] > vol_ma_6[i] * 1.5  # 1.5x average volume for confirmation
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Camarilla R4 AND price > EMA40 AND volume confirmation
            if price > r4_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below Camarilla S4 AND price < EMA40 AND volume confirmation
            elif price < s4_val and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC: reverse on opposite signal ===
        elif position == 1:
            # Exit long if price breaks below Camarilla S3 (we approximate S3 as S4 + 0.5*(R4-S4))
            s3_val = s4_val + 0.5 * (r4_val - s4_val)
            if price < s3_val:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above Camarilla R3 (we approximate R3 as R4 - 0.5*(R4-S4))
            r3_val = r4_val - 0.5 * (r4_val - s4_val)
            if price > r3_val:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_CamarillaR4S4_1wEMA40_Volume1.5x"
timeframe = "6h"
leverage = 1.0