#/usr/bin/env python3
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
    
    # === 12h data for indicators ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h EMA(34) ===
    ema_34 = np.zeros(len(close_12h))
    alpha = 2 / (34 + 1)
    ema_34[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema_34[i] = alpha * close_12h[i] + (1 - alpha) * ema_34[i-1]
    
    # === 12h RSI(14) ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(len(gain))
    avg_loss = np.zeros(len(loss))
    if len(gain) > 0:
        avg_gain[0] = gain[0]
    if len(loss) > 0:
        avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h Volume MA(20) ===
    vol_ma_20_12h = np.zeros(len(volume_12h))
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
        else:
            vol_ma_20_12h[i] = np.mean(volume_12h[max(0, i-9):i+1]) if i > 0 else volume_12h[0]
    
    # Align to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_12h_aligned[i] > vol_ma_20_12h_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA34 and RSI < 35 with volume confirmation
            if close[i] > ema_34_aligned[i] and rsi_12h_aligned[i] < 35 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA34 and RSI > 65 with volume confirmation
            elif close[i] < ema_34_aligned[i] and rsi_12h_aligned[i] > 65 and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral (40-60 range)
        elif position == 1:
            # Exit long: RSI >= 40
            if rsi_12h_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI <= 60
            if rsi_12h_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA34_RSI_Volume_12h"
timeframe = "4h"
leverage = 1.0