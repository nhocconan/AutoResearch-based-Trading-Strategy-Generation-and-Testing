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
    
    # Get 1d data for ATR and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 14-day RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align ATR and RSI to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4-hour EMA50 for trend filter
    ema_period = 50
    ema_4h = np.full(n, np.nan)
    if n >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema_4h[i] = (close[i] * (2 / (ema_period + 1)) + 
                         ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate volume MA for volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_period:
        vol_ma[vol_ma_period - 1] = np.mean(volume[:vol_ma_period])
        for i in range(vol_ma_period, n):
            vol_ma[i] = (volume[i] * (2 / (vol_ma_period + 1)) + 
                         vol_ma[i-1] * (1 - (2 / (vol_ma_period + 1))))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, RSI, EMA, and volume MA
    start_idx = max(14, ema_period - 1, vol_ma_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        ema_trend = ema_4h[i]
        vol_current = volume[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: require volume above average
        volume_confirmed = vol_current > vol_avg
        
        if position == 0:
            # Long: Oversold RSI with price above EMA and volume confirmation
            if (rsi < 30 and price > ema_trend and volume_confirmed):
                signals[i] = size
                position = 1
            # Short: Overbought RSI with price below EMA and volume confirmation
            elif (rsi > 70 and price < ema_trend and volume_confirmed):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or trend fails
            if rsi > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral or trend fails
            if rsi < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_RSI_MeanReversion_1D_ATR_RSI_EMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0