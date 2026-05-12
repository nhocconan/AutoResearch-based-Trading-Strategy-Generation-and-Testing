#!/usr/bin/env python3
# 1h Camarilla Pivot + Volume Spike + 4h Trend
# Hypothesis: Camarilla pivot levels provide high-probability reversal points in ranging markets.
# Combines with 4h EMA trend filter and volume spikes to avoid false breakouts.
# Works in both bull/bear by fading extremes in range and following trend when strong.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "1h_Camarilla_Volume_4hTrend"
timeframe = "1h"
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
    
    # === 4h Data for EMA Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === Daily Data for Camarilla Pivot Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_h4 = typical_price + 1.1 * range_1d / 2.0
    camarilla_l4 = typical_price - 1.1 * range_1d / 2.0
    camarilla_h3 = typical_price + 1.1 * range_1d / 4.0
    camarilla_l3 = typical_price - 1.1 * range_1d / 4.0
    
    # Align to 1h (previous day's levels available at 00:00 UTC)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === Volume Spike (24-period on 1h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at L3/L4 support + volume spike + above 4h EMA20
            if (close[i] <= l3_aligned[i] * 1.001 and  # Allow small buffer
                vol_spike[i] and
                close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price at H3/H4 resistance + volume spike + below 4h EMA20
            elif (close[i] >= h3_aligned[i] * 0.999 and  # Allow small buffer
                  vol_spike[i] and
                  close[i] < ema_20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches H3 or trend weakens
            if close[i] >= h3_aligned[i] * 0.999 or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reaches L3 or trend weakens
            if close[i] <= l3_aligned[i] * 1.001 or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals