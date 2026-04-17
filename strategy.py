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
    
    # === 1d EMA(34) for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Wilder's smoothing EMA (alpha = 1/period)
    alpha = 1.0 / 34
    ema_34 = np.full_like(close_1d, np.nan)
    ema_34[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    
    # === 1d ATR(14) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 1d volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_ma_10 = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_1d)):
        if i >= 9:
            vol_ma_10[i] = np.mean(vol_1d[i-9:i+1])
        elif i > 0:
            vol_ma_10[i] = np.mean(vol_1d[max(0, i-4):i+1])
        else:
            vol_ma_10[i] = vol_1d[0]
    vol_spike = vol_1d > vol_ma_10 * 2.0  # Volume > 2x 10-period average
    
    # Align indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    warmup = 100
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        if i >= 19:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
        vol_confirm = volume[i] > vol_ma_20 * 1.5
        
        # Entry logic: only enter when flat AND volume spike present (1d)
        if position == 0:
            # Long: price above EMA34 + volatility filter + volume confirmation + 1d volume spike
            if (close[i] > ema_34_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm and
                vol_spike_aligned[i] > 0.5):  # 1d volume spike
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA34 + volatility filter + volume confirmation + 1d volume spike
            elif (close[i] < ema_34_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm and
                  vol_spike_aligned[i] > 0.5):  # 1d volume spike
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "EMA34_ATR_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0