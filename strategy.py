#!/usr/bin/env python3
# 4h_12h_camarilla_volume_v1
# Strategy: 4h Camarilla pivot levels with 12h volume confirmation and 12h EMA trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability reversal zones. Combined with 12h volume surge (>1.5x average) and 12h EMA50 trend filter, it captures institutional interest at key levels while avoiding false signals in low-volume or counter-trend environments. Works in both bull and bear markets by fading extremes with confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close
    close_val = close
    L3 = close_val + (range_val * 1.1 / 12)
    L4 = close_val + (range_val * 1.1 / 6)
    H3 = close_val - (range_val * 1.1 / 12)
    H4 = close_val - (range_val * 1.1 / 6)
    return L3, L4, H3, H4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume confirmation: volume > 1.5x 20-period average
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean()
    vol_ratio_12h = pd.Series(vol_12h) / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h.values)
    
    # Calculate Camarilla levels for each 4h bar using current bar's H/L/C
    L3 = np.full(n, np.nan)
    L4 = np.full(n, np.nan)
    H3 = np.full(n, np.nan)
    H4 = np.full(n, np.nan)
    
    for i in range(n):
        L3[i], L4[i], H3[i], H4[i] = calculate_camarilla(high[i], low[i], close[i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(L3[i]) or np.isnan(L4[i]) or np.isnan(H3[i]) or np.isnan(H4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: 12h volume > 1.5x average
        vol_confirmed = vol_ratio_12h_aligned[i] > 1.5
        
        # Entry conditions
        # Long: Price touches L3/L4 support in uptrend with volume confirmation
        if vol_confirmed and (close[i] <= L3[i] or close[i] <= L4[i]) and close[i] > ema_50_12h_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price touches H3/H4 resistance in downtrend with volume confirmation
        elif vol_confirmed and (close[i] >= H3[i] or close[i] >= H4[i]) and close[i] < ema_50_12h_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price moves back toward midpoint or trend fails
        elif position == 1 and (close[i] >= (L3[i] + H3[i]) / 2 or close[i] < ema_50_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= (H3[i] + L3[i]) / 2 or close[i] > ema_50_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals