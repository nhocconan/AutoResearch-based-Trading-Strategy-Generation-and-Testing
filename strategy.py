#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels (high, low, close of previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using previous day's HLC
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        range_hl = h - l
        camarilla_r4[i] = c + (range_hl * 1.1 / 2)
        camarilla_r3[i] = c + (range_hl * 1.1 / 4)
        camarilla_s3[i] = c - (range_hl * 1.1 / 4)
        camarilla_s4[i] = c - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 4h trend filter: EMA(34) on 4h close
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_34 = np.full(len(df_4h), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_4h)):
        if i < 33:
            ema_4h_34[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_4h_34[i-1]):
                ema_4h_34[i] = np.mean(close_4h[i-33:i+1])
            else:
                ema_4h_34[i] = close_4h[i] * alpha + ema_4h_34[i-1] * (1 - alpha)
    
    ema_4h_34_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_34)
    
    # Calculate 4h ATR(14) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_4h = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_4h[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 34)  # volume MA needs 20, 4h EMA needs 34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_4h_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume (strict to reduce trades)
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price touches or breaks above S3 with volume and 4h uptrend
            if (volume_confirmation and 
                price >= camarilla_s3_aligned[i] and 
                close[i-1] < camarilla_s3_aligned[i] and  # just touched/broke
                ema_4h_34_aligned[i] > ema_4h_34_aligned[i-1]):  # 4h uptrend
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks below R3 with volume and 4h downtrend
            elif (volume_confirmation and 
                  price <= camarilla_r3_aligned[i] and 
                  close[i-1] > camarilla_r3_aligned[i] and  # just touched/broke
                  ema_4h_34_aligned[i] < ema_4h_34_aligned[i-1]):  # 4h downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R3 or 4h trend turns down
            if (price >= camarilla_r3_aligned[i] or 
                ema_4h_34_aligned[i] < ema_4h_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price reaches S3 or 4h trend turns up
            if (price <= camarilla_s3_aligned[i] or 
                ema_4h_34_aligned[i] > ema_4h_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Camarilla_S3R3_4hEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0