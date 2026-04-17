#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d RSI(14) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI: 100 - (100 / (1 + RS))
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    period = 14
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    rsi = np.full_like(close_1d, np.nan)
    
    # Seed with simple average
    if len(close_1d) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        if avg_loss[period-1] != 0:
            rs = avg_gain[period-1] / avg_loss[period-1]
            rsi[period-1] = 100 - (100 / (1 + rs))
        else:
            rsi[period-1] = 100 if avg_gain[period-1] > 0 else 50
        
        # Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(close_1d)):
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100 if avg_gain[i] > 0 else 50
    
    # === 1d EMA(50) for trend filter ===
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # seed
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    else:
        for i in range(len(close_1d)):
            ema_50[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === Align indicators to 4h timeframe ===
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 4h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === RSI levels ===
    OVERBOUGHT = 70
    OVERSOLD = 30
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI crosses above 30 from below AND price above EMA50
            if (rsi_aligned[i] > OVERSOLD and 
                rsi_aligned[i-1] <= OVERSOLD and  # crossed up
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI crosses below 70 from above AND price below EMA50
            elif (rsi_aligned[i] < OVERBOUGHT and 
                  rsi_aligned[i-1] >= OVERBOUGHT and  # crossed down
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses below 50 OR crosses above 70
            if (rsi_aligned[i] < 50 and rsi_aligned[i-1] >= 50) or \
               (rsi_aligned[i] < OVERBOUGHT and rsi_aligned[i-1] >= OVERBOUGHT):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 OR crosses below 30
            if (rsi_aligned[i] > 50 and rsi_aligned[i-1] <= 50) or \
               (rsi_aligned[i] > OVERSOLD and rsi_aligned[i-1] <= OVERSOLD):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_EMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0