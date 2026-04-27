#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for calculations (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4-hour RSI (14-period)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_4h), np.nan)
    avg_loss = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(close_4h)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_14_4h = np.full(len(close_4h), np.nan)
    for i in range(14, len(close_4h)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_14_4h[i] = 100 - (100 / (1 + rs))
    
    # Calculate 4-hour EMA (34-period)
    ema_34_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 34:
        alpha = 2 / (34 + 1)
        ema_34_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema_34_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_34_4h[i-1]
    
    # Calculate 4-period volume average for spike detection
    vol_ma_4h = np.full(len(close_4h), np.nan)
    vol_period = 4
    for i in range(vol_period, len(close_4h)):
        vol_ma_4h[i] = np.mean(volume[i-vol_period:i])
    
    # Align 4h indicators to 1h timeframe
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Get 1h data for entry timing
    vol_ma_1h = np.full(n, np.nan)
    vol_period_1h = 4
    for i in range(vol_period_1h, n):
        vol_ma_1h[i] = np.mean(volume[i-vol_period_1h:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio_1h = volume[i] / vol_ma_1h[i] if vol_ma_1h[i] > 0 else 0
        vol_ratio_4h = volume[i] / vol_ma_4h_aligned[i] if vol_ma_4h_aligned[i] > 0 else 0
        
        # Volume filter: at least 1.5x average volume on both timeframes
        vol_filter = vol_ratio_1h > 1.5 and vol_ratio_4h > 1.5
        
        if position == 0:
            # Long: RSI oversold (<30) + price above EMA34 + volume spike
            if rsi_14_4h_aligned[i] < 30 and price > ema_34_4h_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) + price below EMA34 + volume spike
            elif rsi_14_4h_aligned[i] > 70 and price < ema_34_4h_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI overbought (>70) or price below EMA34
            if rsi_14_4h_aligned[i] > 70 or price < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI oversold (<30) or price above EMA34
            if rsi_14_4h_aligned[i] < 30 or price > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI14_EMA34_4h_Volume"
timeframe = "1h"
leverage = 1.0