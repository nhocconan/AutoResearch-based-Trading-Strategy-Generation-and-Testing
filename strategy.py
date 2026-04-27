#!/usr/bin/env python3
"""
4h_RSI_Contrarian_With_Volume_and_Trend_Filter
RSI-based mean reversion with volume confirmation and multi-timeframe trend filter.
Long when RSI < 30 (oversold) + volume spike + 12h EMA50 uptrend.
Short when RSI > 70 (overbought) + volume spike + 12h EMA50 downtrend.
Exit when RSI returns to neutral range (40-60) or trend fails.
Position size: 0.25. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < rsi_period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
                avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period + 1:i + 1])
    volume_spike = volume > (vol_ma * 1.5)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                          ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, volume MA, and EMA
    start_idx = max(rsi_period, vol_ma_period - 1, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        ema12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + volume spike + price above 12h EMA50 (uptrend)
            if (rsi_val < 30 and vol_spike and price > ema12h_val):
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) + volume spike + price below 12h EMA50 (downtrend)
            elif (rsi_val > 70 and vol_spike and price < ema12h_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or trend fails
            if (rsi_val >= 40 and rsi_val <= 60) or price < ema12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or trend fails
            if (rsi_val >= 40 and rsi_val <= 60) or price > ema12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI_Contrarian_With_Volume_and_Trend_Filter"
timeframe = "4h"
leverage = 1.0