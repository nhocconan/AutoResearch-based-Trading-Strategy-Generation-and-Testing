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
    
    # === 4h EMA34 (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d RSI (14) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1d Volume (average over last 5 days) ===
    vol_1d = df_1d['volume'].values
    vol_ma_5 = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_1d)):
        if i >= 4:
            vol_ma_5[i] = np.mean(vol_1d[i-4:i+1])
        elif i > 0:
            vol_ma_5[i] = np.mean(vol_1d[max(0, i-2):i+1])
        else:
            vol_ma_5[i] = vol_1d[0]
    vol_ma_5_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_5)
    
    # === 1h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    warmup = 100
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_5_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 5-day average 1d volume
        vol_confirm = volume[i] > vol_ma_5_aligned[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA34(4h) + RSI < 40 + volume confirmation
            if (close[i] > ema_34_4h_aligned[i] and 
                rsi_1d_aligned[i] < 40 and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA34(4h) + RSI > 60 + volume confirmation
            elif (close[i] < ema_34_4h_aligned[i] and 
                  rsi_1d_aligned[i] > 60 and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 50 OR price closes below EMA34(4h)
            if (rsi_1d_aligned[i] > 50 or 
                close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 OR price closes above EMA34(4h)
            if (rsi_1d_aligned[i] < 50 or 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA34_RSI14_VolFilter_v1"
timeframe = "1h"
leverage = 1.0