#!/usr/bin/env python3
"""
4h_1d_Volume_Weighted_RSI_Momentum
Hypothesis: Combine 1-hour RSI momentum with volume-weighted price action and 1-day trend filter.
- Long: RSI(14) > 55 on 1h (derived from 4h), price > VWAP(20), and close > 1-day EMA(50)
- Short: RSI(14) < 45 on 1h, price < VWAP(20), and close < 1-day EMA(50)
- Volume confirmation: current volume > 1.3 x 20-period average
- Uses discrete position sizing (0.25) to minimize fee churn
- Designed for 4h timeframe with 1d trend filter to reduce whipsaws and capture medium-term momentum
- Targets 20-35 trades/year by requiring multiple confluence factors
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA(50) for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = close_1d[:50].mean()
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 0.0392) + (ema_50_1d[i-1] * 0.9608)
    
    # Align 1-day EMA(50) to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1h data for RSI (approximated from 4h by using every 4th bar)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate RSI(14) on 1h data
    rsi_1h = np.full_like(close_1h, np.nan)
    if len(close_1h) >= 15:
        delta = np.diff(close_1h)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1h, np.nan)
        avg_loss = np.full_like(close_1h, np.nan)
        
        avg_gain[14] = gain[:14].mean()
        avg_loss[14] = loss[:14].mean()
        
        for i in range(15, len(close_1h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1h = 100 - (100 / (1 + rs))
    
    # Align 1h RSI to 4h timeframe (each 4h bar contains 4 1h bars)
    # We'll use the last 1h bar's RSI value for each 4h period
    rsi_1h_aligned = np.full(n, np.nan)
    for i in range(n):
        # Map 4h bar index to 1h bar index (4 1h bars per 4h bar)
        idx_1h = i * 4 + 3  # Last 1h bar in the 4h period
        if idx_1h < len(rsi_1h):
            rsi_1h_aligned[i] = rsi_1h[idx_1h]
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    # VWAP(20) approximation for 4h
    vwap = np.full(n, np.nan)
    for i in range(20, n):
        typical_price = (high[i-20:i] + low[i-20:i] + close[i-20:i]) / 3
        vwap[i] = np.dot(typical_price, volume[i-20:i]) / volume[i-20:i].sum()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need VWAP(20), volume MA, and RSI(14) with 4-hour offset
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI > 55, price > VWAP, close > 1-day EMA(50), with volume
            if (rsi_1h_aligned[i] > 55 and close[i] > vwap[i] and 
                close[i] > ema_50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI < 45, price < VWAP, close < 1-day EMA(50), with volume
            elif (rsi_1h_aligned[i] < 45 and close[i] < vwap[i] and 
                  close[i] < ema_50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI < 50 or price < VWAP (momentum fade)
            if (rsi_1h_aligned[i] < 50 or close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI > 50 or price > VWAP (momentum fade)
            if (rsi_1h_aligned[i] > 50 or close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Volume_Weighted_RSI_Momentum"
timeframe = "4h"
leverage = 1.0