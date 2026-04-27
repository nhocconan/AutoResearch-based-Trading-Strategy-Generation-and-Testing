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
    
    # Get 12h data for primary HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(34) on 12h close
    ema_12h_34 = np.full(len(df_12h), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_12h)):
        if i < 33:
            ema_12h_34[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_12h_34[i-1]):
                ema_12h_34[i] = np.mean(close_12h[i-33:i+1])
            else:
                ema_12h_34[i] = close_12h[i] * alpha + ema_12h_34[i-1] * (1 - alpha)
    
    ema_12h_34_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_34)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate volume SMA(20) on 1d volume
    vol_sma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(len(volume_1d)):
        if i < 19:
            vol_sma_20_1d[i] = np.mean(volume_1d[:i+1]) if i > 0 else volume_1d[i]
        else:
            vol_sma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Calculate 4h volume ratio: current volume / 12-period average
    vol_ma_12 = np.full(n, np.nan)
    for i in range(n):
        if i < 11:
            vol_ma_12[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_ma_12[i] = np.mean(volume[i-11:i+1])
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma_12)) & (vol_ma_12 > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma_12[valid_vol]
    
    # Calculate 4-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(4, n):
        if i == 4:
            avg_gain[i] = np.mean(gain[1:5])
            avg_loss[i] = np.mean(loss[1:5])
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_4 = np.full(n, np.nan)
    rsi_4[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(34, 20, 4)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_34_aligned[i]) or 
            np.isnan(volume_ratio[i]) or
            np.isnan(rsi_4[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike filter: current volume > 1.5x 12-period average
        volume_spike = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volume spike + 12h uptrend
            if (rsi_4[i] < 30 and 
                volume_spike and 
                ema_12h_34_aligned[i] > ema_12h_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volume spike + 12h downtrend
            elif (rsi_4[i] > 70 and 
                  volume_spike and 
                  ema_12h_34_aligned[i] < ema_12h_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or 12h trend turns down
            if (rsi_4[i] > 70 or 
                ema_12h_34_aligned[i] < ema_12h_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or 12h trend turns up
            if (rsi_4[i] < 30 or 
                ema_12h_34_aligned[i] > ema_12h_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_RSI4_12hEMA34_v1"
timeframe = "4h"
leverage = 1.0