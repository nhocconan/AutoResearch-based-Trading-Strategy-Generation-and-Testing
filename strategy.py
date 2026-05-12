#!/usr/bin/env python3
# 4h_Price_Action_With_1dVWAP_and_Volume
# Hypothesis: Price returning to the previous day's VWAP acts as a mean-reversion signal in ranging markets, while breaks above/below VWAP with volume and trend alignment indicate momentum in trending markets. This dual-regime approach works in both bull and bear markets by adapting to price action around the 1-day VWAP, filtered by 1-week EMA trend and volume confirmation. Designed for low trade frequency and high edge.

name = "4h_Price_Action_With_1dVWAP_and_Volume"
timeframe = "4h"
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
    
    # === 1-week EMA for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 1-day VWAP ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d_array = vwap_1d.values
    
    # Align VWAP to 4h
    vwap_4h = align_htf_to_ltf(prices, df_1d, vwap_1d_array)
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h[i]) or np.isnan(vwap_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above VWAP with volume and uptrend
            if close[i] > vwap_4h[i] and close[i-1] <= vwap_4h[i-1] and ema_20_4h[i] > ema_20_4h[i-1] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below VWAP with volume and downtrend
            elif close[i] < vwap_4h[i] and close[i-1] >= vwap_4h[i-1] and ema_20_4h[i] < ema_20_4h[i-1] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below VWAP or trend turns down
            if close[i] < vwap_4h[i] and close[i-1] >= vwap_4h[i-1] or ema_20_4h[i] < ema_20_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above VWAP or trend turns up
            if close[i] > vwap_4h[i] and close[i-1] <= vwap_4h[i-1] or ema_20_4h[i] > ema_20_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals