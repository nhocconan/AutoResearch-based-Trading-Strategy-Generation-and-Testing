#!/usr/bin/env python3
name = "6h_HeikinAshi_ElderRay_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Heikin-Ashi calculation (vectorized) ---
    ha_close = (high + low + close + prices['open'].values) / 4
    ha_open = np.zeros_like(close)
    ha_open[0] = (prices['open'].values[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # --- Elder Ray Power (13-period EMA) ---
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # --- Weekly trend filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Ensure warmup for EMA and HA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: HA bullish, bull power positive, above weekly EMA50, volume ok
        if (ha_close[i] > ha_open[i] and 
            bull_power[i] > 0 and 
            close[i] > ema50_1w_aligned[i] and 
            vol_ok[i]):
            if position <= 0:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short conditions: HA bearish, bear power negative, below weekly EMA50, volume ok
        elif (ha_close[i] < ha_open[i] and 
              bear_power[i] < 0 and 
              close[i] < ema50_1w_aligned[i] and 
              vol_ok[i]):
            if position >= 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: HA color change or loss of power
        else:
            if position == 1 and ha_close[i] < ha_open[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and ha_close[i] > ha_open[i]:
                signals[i] = 0.0
                position = 0
            elif position == 1 and bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals