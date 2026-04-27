#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla S3/R3 breakout with volume confirmation and weekly EMA trend filter
# Works in bull/bear by using weekly EMA to filter trend direction, reducing false signals in volatile markets
# Target: 50-150 trades over 4 years (12-37/year) - conservative to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need at least 34 bars for weekly EMA warmup
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: S3, R3 based on previous day's HLC
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        range_hl = h - l
        camarilla_s3[i] = c - (range_hl * 1.1 / 4)
        camarilla_r3[i] = c + (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # Get weekly trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = np.full(len(df_1w), np.nan)
    # Calculate EMA with proper initialization
    ema_1w_34[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        alpha = 2 / (34 + 1)
        ema_1w_34[i] = close_1w[i] * alpha + ema_1w_34[i-1] * (1 - alpha)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate 12h ATR(14) for dynamic position sizing (optional risk control)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_12h = np.zeros(n)
    atr_12h[0] = tr[0]
    for i in range(1, n):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period) with min_periods
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need volume MA (20) and weekly EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(ema_1w_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume (strict to reduce trades)
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price touches or breaks above S3 with volume and weekly uptrend
            if (volume_confirmation and 
                price >= camarilla_s3_aligned[i] and 
                close[i-1] < camarilla_s3_aligned[i] and  # just touched/broke
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):  # weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks below R3 with volume and weekly downtrend
            elif (volume_confirmation and 
                  price <= camarilla_r3_aligned[i] and 
                  close[i-1] > camarilla_r3_aligned[i] and  # just touched/broke
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):  # weekly downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R3 or weekly trend turns down
            if (price >= camarilla_r3_aligned[i] or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price reaches S3 or weekly trend turns up
            if (price <= camarilla_s3_aligned[i] or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_Camarilla_S3R3_WeeklyEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0