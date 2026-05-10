#!/usr/bin/env python3
"""
6h_RSI2_TrendFilter_1dVWAP
Hypothesis: 2-period RSI with 1d VWAP trend filter and volume confirmation.
RSI2 is sensitive to short-term reversals; filtered by 1d VWAP trend to avoid counter-trend trades.
In bull markets, buy dips above VWAP; in bear markets, sell rallies below VWAP.
Volume confirmation ensures institutional participation. Target: 50-150 trades over 4 years.
"""

name = "6h_RSI2_TrendFilter_1dVWAP"
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
    
    # 1d VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    volume_1d = df_1d['volume'].values
    
    vwap_1d = np.full(len(typical_price_1d), np.nan)
    cumulative_tpv = 0.0
    cumulative_volume = 0.0
    for i in range(len(typical_price_1d)):
        cumulative_tpv += typical_price_1d[i] * volume_1d[i]
        cumulative_volume += volume_1d[i]
        if cumulative_volume > 0:
            vwap_1d[i] = cumulative_tpv / cumulative_volume
    
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 2-period RSI on 6h data
    rsi2 = np.full(n, np.nan)
    if n >= 2:
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        # Wilder's smoothing for RSI
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        if n >= 2:
            avg_gain[1] = np.mean(gain[:2])
            avg_loss[1] = np.mean(loss[:2])
            
            # Wilder smoothing
            for i in range(2, n):
                avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
                avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi2 = 100 - (100 / (1 + rs))
        # Handle division by zero
        rsi2 = np.where(avg_loss == 0, 100, rsi2)
        rsi2 = np.where(avg_gain == 0, 0, rsi2)
    
    # Volume confirmation: 6h volume > 1.5x average 6h volume from 1d
    # Approximate average 6h volume from 1d: volume_1d / 4
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    vol_6h_avg_approx = vol_sma20_1d_aligned / 4.0
    volume_confirm = volume > 1.5 * vol_6h_avg_approx
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi2[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI2 < 15 (oversold) and price above VWAP (uptrend) with volume
            if rsi2[i] < 15 and close[i] > vwap_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 85 (overbought) and price below VWAP (downtrend) with volume
            elif rsi2[i] > 85 and close[i] < vwap_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI2 > 70 (overbought) or price below VWAP (trend change)
            if rsi2[i] > 70 or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI2 < 30 (oversold) or price above VWAP (trend change)
            if rsi2[i] < 30 or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals