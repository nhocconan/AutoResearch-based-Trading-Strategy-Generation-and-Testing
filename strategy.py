#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d RSI (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing with min_periods
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === 1w trend: EMA50 ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i == 0:
            ema_50[i] = close_1w[i]
        else:
            ema_50[i] = (close_1w[i] * 2 + ema_50[i-1] * 49) / 51
    
    # === Align indicators to 1d timeframe ===
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # === 1d Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: RSI < 30 (oversold) + price above weekly EMA50 + volume confirmation
            if (rsi_1d_aligned[i] < 30 and 
                close[i] > ema_50_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70 (overbought) + price below weekly EMA50 + volume confirmation
            elif (rsi_1d_aligned[i] > 70 and 
                  close[i] < ema_50_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI14_WeeklyTrend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0