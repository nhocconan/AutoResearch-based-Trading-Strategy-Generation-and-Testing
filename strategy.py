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
    
    # === Daily ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
    atr = np.full_like(tr, np.nan)
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr[i] = tr[i]
            else:
                atr[i] = (atr[i-1] * (i-1) + tr[i]) / i
        else:
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    atr_ma_50 = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if i >= 49:
            atr_ma_50[i] = np.mean(atr[i-49:i+1])
        elif i > 0:
            atr_ma_50[i] = np.mean(atr[max(0, i-24):i+1])
        else:
            atr_ma_50[i] = atr[0]
    
    # Volatility filter: ATR < 50-period MA (low volatility regime)
    vol_filter = atr < atr_ma_50
    
    # === Daily Donchian Channel (20-period) ===
    donch_high = np.full_like(close_1d, np.nan)
    donch_low = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            donch_high[i] = np.max(high_1d[i-19:i+1])
            donch_low[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            donch_high[i] = np.max(high_1d[max(0, i-9):i+1])
            donch_low[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            donch_high[i] = high_1d[0]
            donch_low[i] = low_1d[0]
    
    # === Daily RSI (14-period) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
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
    
    # Align indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    # === Session filter (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_filter_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high + RSI < 50 + low vol + volume confirmation
            if (close[i] > donch_high_aligned[i] and 
                rsi_1d_aligned[i] < 50 and 
                vol_filter_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low + RSI > 50 + low vol + volume confirmation
            elif (close[i] < donch_low_aligned[i] and 
                  rsi_1d_aligned[i] > 50 and 
                  vol_filter_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian low OR RSI > 70
            if (close[i] < donch_low_aligned[i] or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high OR RSI < 30
            if (close[i] > donch_high_aligned[i] or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_RSI_VolFilter_Session_v1"
timeframe = "4h"
leverage = 1.0