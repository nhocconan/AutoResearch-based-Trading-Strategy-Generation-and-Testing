#!/usr/bin/env python3
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
    
    # === 1w EMA50 ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA50 on weekly
    ema_50 = np.zeros(len(close_1w))
    alpha = 2 / (50 + 1)
    ema_50[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_50[i] = alpha * close_1w[i] + (1 - alpha) * ema_50[i-1]
    
    # === 1w RSI(14) ===
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    avg_gain[0] = gain[0] if len(gain) > 0 else 0
    avg_loss[0] = loss[0] if len(loss) > 0 else 0
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # === 1w Volume MA(10) ===
    vol_ma_10_1w = np.zeros(len(volume_1w))
    for i in range(len(volume_1w)):
        if i >= 9:
            vol_ma_10_1w[i] = np.mean(volume_1w[i-9:i+1])
        else:
            vol_ma_10_1w[i] = np.mean(volume_1w[max(0, i-4):i+1]) if i > 0 else volume_1w[0]
    
    # Align to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma_10_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current weekly volume > 1.3x 10-period average
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        vol_confirm = vol_1w_aligned[i] > vol_ma_10_1w_aligned[i] * 1.3
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above weekly EMA50 and RSI < 30 with volume confirmation
            if close[i] > ema_50_aligned[i] and rsi_1w_aligned[i] < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below weekly EMA50 and RSI > 70 with volume confirmation
            elif close[i] < ema_50_aligned[i] and rsi_1w_aligned[i] > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral range
        elif position == 1:
            # Exit long: RSI >= 40
            if rsi_1w_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI <= 60
            if rsi_1w_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA50_RSI_Volume"
timeframe = "1d"
leverage = 1.0