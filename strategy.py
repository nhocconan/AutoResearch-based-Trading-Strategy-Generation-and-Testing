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
    
    # Get weekly data for calculations (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly close for Donchian channel
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    upper = np.full(len(close_1w), np.nan)
    lower = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        for i in range(20-1, len(close_1w)):
            upper[i] = np.max(close_1w[i-20+1:i+1])
            lower[i] = np.min(close_1w[i-20+1:i+1])
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = np.roll(close_1w, 1)
    close_1w_prev[0] = close_1w[0]
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w_prev)
    tr3 = np.abs(low_1w - close_1w_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1w = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1w[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1w[i] = (atr_14_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly EMA (50-period) for trend filter
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align weekly indicators to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 5-day volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 5
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above EMA50 trend
            if price > upper_1w_aligned[i] and vol_filter and price > ema_50_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian with volume and below EMA50 trend
            elif price < lower_1w_aligned[i] and vol_filter and price < ema_50_1w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower Donchian or volatility spike (potential reversal)
            if price < lower_1w_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper Donchian or volatility spike (potential reversal)
            if price > upper_1w_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian_20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0