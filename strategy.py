#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper/lower bands
    donch_high = np.full_like(high_12h, np.nan)
    donch_low = np.full_like(low_12h, np.nan)
    period = 20
    for i in range(len(high_12h)):
        if i >= period - 1:
            donch_high[i] = np.max(high_12h[i-period+1:i+1])
            donch_low[i] = np.min(low_12h[i-period+1:i+1])
        else:
            donch_high[i] = high_12h[0]
            donch_low[i] = low_12h[0]
    
    # === 1d Volume Confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d timeframe
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= period - 1:
            vol_ma_20[i] = np.mean(volume_1d[i-period+1:i+1])
        else:
            vol_ma_20[i] = volume_1d[0]
    
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    vol_confirm = volume_1d > vol_ma_20 * 1.5
    
    # === 1d RSI (14-period) for momentum filter ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period_rsi = 14
    for i in range(len(gain)):
        if i < period_rsi:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period_rsi-1) + gain[i]) / period_rsi
            avg_loss[i] = (avg_loss[i-1] * (period_rsi-1) + loss[i]) / period_rsi
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === Align indicators to main timeframe ===
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Break above Donchian high + volume confirmation + RSI > 50
            if (close[i] > donch_high_aligned[i] and 
                vol_confirm_aligned[i] and 
                rsi_1d_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Break below Donchian low + volume confirmation + RSI < 50
            elif (close[i] < donch_low_aligned[i] and 
                  vol_confirm_aligned[i] and 
                  rsi_1d_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Close below Donchian low OR RSI < 40
            if (close[i] < donch_low_aligned[i] or 
                rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close above Donchian high OR RSI > 60
            if (close[i] > donch_high_aligned[i] or 
                rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0