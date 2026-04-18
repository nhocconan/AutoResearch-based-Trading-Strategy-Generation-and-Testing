#!/usr/bin/env python3
"""
4h_RSI_Trend_Momentum_Volume
Hypothesis: Combines RSI(2) for momentum with 1d EMA50 trend filter and volume confirmation. RSI(2) captures short-term reversals in both bull and bear markets, while the 1d EMA50 ensures alignment with the higher-timeframe trend. Volume confirmation filters out low-conviction moves. Designed for moderate trade frequency (~20-40 trades/year) to avoid fee drag.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    if len(close_1d) >= 50:
        k = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * k + ema_50_1d[i-1] * (1 - k)
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(2) on close
    rsi_period = 2
    rsi = np.zeros_like(close)
    rsi[:] = np.nan
    if len(close) >= rsi_period:
        change = np.diff(close, prepend=close[0])
        gain = np.where(change > 0, change, 0.0)
        loss = np.where(change < 0, -change, 0.0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, rsi_period)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: RSI(2) < 15 (oversold) + above 1d EMA50 + volume spike
            if rsi[i] < 15 and close[i] > ema_50_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: RSI(2) > 85 (overbought) + below 1d EMA50 + volume spike
            elif rsi[i] > 85 and close[i] < ema_50_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: RSI(2) > 60 (overbought) or below 1d EMA50 or min 4 bars held
            if bars_since_entry >= 4:
                if rsi[i] > 60 or close[i] < ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: RSI(2) < 40 (oversold) or above 1d EMA50 or min 4 bars held
            if bars_since_entry >= 4:
                if rsi[i] < 40 or close[i] > ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "4h_RSI_Trend_Momentum_Volume"
timeframe = "4h"
leverage = 1.0