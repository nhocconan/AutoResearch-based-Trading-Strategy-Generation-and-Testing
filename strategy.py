#!/usr/bin/env python3
"""
6h_1d_200ema_1w_vwap_reversion
Hypothesis: Mean reversion on 6-hour timeframe when price deviates significantly from weekly VWAP, filtered by daily 200 EMA trend direction. Works in both bull and bear markets by fading extremes only when aligned with higher timeframe trend. Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get weekly data for VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Typical price for VWAP calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Cumulative values for VWAP
    cum_vol = np.cumsum(volume_1w)
    cum_vol_tp = np.cumsum(volume_1w * typical_price_1w)
    
    # VWAP calculation (avoid division by zero)
    vwap_1w = np.divide(cum_vol_tp, cum_vol, out=np.full_like(cum_vol_tp, np.nan), where=cum_vol!=0)
    
    # Align weekly VWAP to 6h timeframe with 1-bar delay (needs next week to confirm)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w, additional_delay_bars=1)
    
    # Calculate 6-period ATR for entry/exit conditions
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend determination: price above/below daily EMA200
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        
        # Deviation from weekly VWAP in ATR units
        if vwap_1w_aligned[i] != 0:
            deviation_atr = abs(close[i] - vwap_1w_aligned[i]) / atr_6h[i]
        else:
            deviation_atr = 0
        
        # Entry conditions: mean reversion when price deviates significantly from VWAP
        # Long when price is significantly below VWAP and above EMA200 (uptrend bias)
        if price_above_ema200 and deviation_atr > 2.5 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short when price is significantly above VWAP and below EMA200 (downtrend bias)
        elif price_below_ema200 and deviation_atr > 2.5 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when price returns to VWAP area
        elif position == 1 and close[i] >= vwap_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= vwap_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_200ema_1w_vwap_reversion"
timeframe = "6h"
leverage = 1.0