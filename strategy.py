#!/usr/bin/env python3
"""
12h_VolumeWeighted_PriceAction_Strategy
Hypothesis: Combine 12h price action with 1d volume-weighted price action to capture multi-day moves in BTC/ETH.
Uses 1d VWAP for trend, 12h price crossing VWAP with volume confirmation for entries, and ATR-based volatility filter.
Targets 20-30 trades/year by requiring VWAP cross + volume spike + volatility filter.
Works in bull/bear via VWAP trend filter + avoids false signals in low volatility.
"""

name = "12h_VolumeWeighted_PriceAction_Strategy"
timeframe = "12h"
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d VWAP (Volume Weighted Average Price)
    # VWAP = sum(price * volume) / sum(volume) for the day
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Calculate 12h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values  # 12-period ATR
    atr_avg_12 = pd.Series(atr_12).rolling(window=24, min_periods=24).mean().values  # 24-period average ATR (2 days)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need VWAP (2) and ATR avg (24)
    start_idx = max(2, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(atr_12[i]) or 
            np.isnan(atr_avg_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d VWAP)
        above_vwap = close[i] > vwap_1d_aligned[i]
        below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Volume filter: current 12h volume > 2.0x average 12h volume
        vol_12h = volume[i]
        vol_avg_12h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values[i]  # 2-day average
        volume_filter = vol_12h > vol_avg_12h * 2.0
        
        # Volatility filter: current ATR > 0.8x average ATR (avoid extremely low volatility)
        vol_filter = atr_12[i] > atr_avg_12[i] * 0.8
        
        if position == 0:
            # Long entry: price above VWAP + volume spike + adequate volatility
            if above_vwap and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price below VWAP + volume spike + adequate volatility
            elif below_vwap and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP or volatility drops significantly
            if not above_vwap or (atr_12[i] < atr_avg_12[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP or volatility drops significantly
            if not below_vwap or (atr_12[i] < atr_avg_12[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals