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
    
    # === 1d Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian upper and lower bands
    upper_don = np.full_like(high_1d, np.nan)
    lower_don = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            upper_don[i] = np.max(high_1d[i-19:i+1])
            lower_don[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            upper_don[i] = np.max(high_1d[max(0, i-9):i+1])
            lower_don[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            upper_don[i] = high_1d[0]
            lower_don[i] = low_1d[0]
    
    # === 1d RSI (14-period) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
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
    
    # === Align indicators to 4h timeframe ===
    upper_don_4h = align_htf_to_ltf(prices, df_1d, upper_don)
    lower_don_4h = align_htf_to_ltf(prices, df_1d, lower_don)
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h timeframe
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    # === 4h Volatility filter (ATR-based) ===
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full_like(tr, np.nan)
    atr_period = 14
    for i in range(len(tr)):
        if i < atr_period:
            if i == 0:
                atr[i] = tr[i]
            else:
                atr[i] = (atr[i-1] * (i-1) + tr[i]) / i
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Volatility filter: ATR > 20-period ATR average (avoid low volatility)
    atr_ma_20 = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if i >= 19:
            atr_ma_20[i] = np.mean(atr[i-19:i+1])
        elif i > 0:
            atr_ma_20[i] = np.mean(atr[max(0, i-9):i+1])
        else:
            atr_ma_20[i] = atr[0]
    vol_filter = atr > atr_ma_20
    
    # === 4h Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_don_4h[i]) or 
            np.isnan(lower_don_4h[i]) or 
            np.isnan(rsi_1d_4h[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session or volatility filter
        if not session_filter[i] or not vol_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Price breaks above Donchian upper + RSI > 50 (bullish bias) + volume confirmation
            if (close[i] > upper_don_4h[i] and 
                rsi_1d_4h[i] > 50 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian lower + RSI < 50 (bearish bias) + volume confirmation
            elif (close[i] < lower_don_4h[i] and 
                  rsi_1d_4h[i] < 50 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price crosses below Donchian lower OR RSI < 40
            if (close[i] < lower_don_4h[i] or 
                rsi_1d_4h[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper OR RSI > 60
            if (close[i] > upper_don_4h[i] or 
                rsi_1d_4h[i] > 60):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_RSI_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0