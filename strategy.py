#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day RSI (14-period) - Wilder's method
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs))
        else:
            rsi_1d[i] = 100
    
    # Calculate 1-day ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-day volume moving average (20-period)
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        for i in range(20, len(volume_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    rsi_1d_aligned = rsi_1d
    atr_14_1d_aligned = atr_14_1d
    vol_ma_20_1d_aligned = vol_ma_20_1d
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(34, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        # RSI in neutral zone: 40-60
        rsi_filter = (rsi_1d_aligned[i] >= 40) & (rsi_1d_aligned[i] <= 60)
        
        if position == 0:
            # Long: RSI < 40 (oversold) with volume spike
            if rsi_1d_aligned[i] < 40 and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI > 60 (overbought) with volume spike
            elif rsi_1d_aligned[i] > 60 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 60 (overbought) or loss of volume momentum
            if rsi_1d_aligned[i] > 60 or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI < 40 (oversold) or loss of volume momentum
            if rsi_1d_aligned[i] < 40 or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_Volume_MeanReversion"
timeframe = "1d"
leverage = 1.0