#!/usr/bin/env python3
"""
6h_OrderFlow_Imbalance_VWAP_Divergence
Hypothesis: Uses 6h VWAP and order flow imbalance (buying vs selling pressure) to detect institutional accumulation/distribution.
Combines with 1d trend filter (EMA50) and volume confirmation to trade in direction of higher timeframe trend.
Works in both bull and bear markets by following institutional flow while avoiding counter-trend whipsaws.
Target: 15-35 trades/year on 6f timeframe.
"""

name = "6h_OrderFlow_Imbalance_VWAP_Divergence"
timeframe = "6h"
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
    
    # === 1d Higher Timeframe Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h VWAP Calculation (Typical Price * Volume) / Cumulative Volume ===
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # === Order Flow Imbalance: Buying Pressure vs Selling Pressure ===
    # Buying pressure: close in upper half of range
    # Selling pressure: close in lower half of range
    range_hl = high - low
    # Avoid division by zero
    buying_pressure = np.divide(close - low, range_hl, out=np.zeros_like(close), where=range_hl!=0)
    selling_pressure = np.divide(high - close, range_hl, out=np.zeros_like(close), where=range_hl!=0)
    
    # Smoothed pressure difference (3-period EMA)
    pressure_diff = buying_pressure - selling_pressure  # -1 to +1
    pressure_ema = pd.Series(pressure_diff).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # === Volume Filter: 1.8x 20-period EMA ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA50 and EMA3)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(pressure_ema[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above VWAP + buying pressure + uptrend + volume spike
            if (close[i] > vwap[i] and 
                pressure_ema[i] > 0.15 and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP + selling pressure + downtrend + volume spike
            elif (close[i] < vwap[i] and 
                  pressure_ema[i] < -0.15 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP OR selling pressure dominates
            if (close[i] < vwap[i] or pressure_ema[i] < -0.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above VWAP OR buying pressure dominates
            if (close[i] > vwap[i] or pressure_ema[i] > 0.1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals