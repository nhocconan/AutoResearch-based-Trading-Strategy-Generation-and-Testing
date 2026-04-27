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
    
    # Get daily data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR ratio: current ATR(7) / ATR(14) - volatility expansion signal
    tr1_7 = high_1d[1:] - low_1d[1:]
    tr2_7 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_7 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_7d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1_7, np.maximum(tr2_7, tr3_7))])
    
    atr_7_1d = np.full(len(df_1d), np.nan)
    for i in range(7, len(tr_7d)):
        atr_7_1d[i] = np.mean(tr_7d[i-7:i])
    
    atr_7_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_7_1d)
    
    # ATR ratio: ATR(7)/ATR(14) > 1.3 indicates volatility expansion
    atr_ratio = np.full(n, np.nan)
    valid_mask = (~np.isnan(atr_7_1d_aligned)) & (~np.isnan(atr_14_1d_aligned)) & (atr_14_1d_aligned > 0)
    atr_ratio[valid_mask] = atr_7_1d_aligned[valid_mask] / atr_14_1d_aligned[valid_mask]
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper and lower bands
    donchian_upper = np.full(len(df_4h), np.nan)
    donchian_lower = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        if i >= 19:
            donchian_upper[i] = np.max(high_4h[i-19:i+1])
            donchian_lower[i] = np.min(low_4h[i-19:i+1])
        else:
            donchian_upper[i] = np.max(high_4h[:i+1])
            donchian_lower[i] = np.min(low_4h[:i+1])
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Get weekly data for trend filter: EMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_1w_50[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_50[i-1]):
                ema_1w_50[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_1w_50[i] = close_1w[i] * alpha_w + ema_1w_50[i-1] * (1 - alpha_w)
    
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_ratio[i]) or 
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_1w_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility regime filter: ATR ratio > 1.3 = expansion (favor trend)
        vol_expansion = atr_ratio[i] > 1.3
        
        if position == 0:
            # Long: Price breaks above Donchian upper + volatility expansion + weekly uptrend
            if (price > donchian_upper_aligned[i] and 
                vol_expansion and 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volatility expansion + weekly downtrend
            elif (price < donchian_lower_aligned[i] and 
                  vol_expansion and 
                  ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian lower or weekly trend turns down
            if (price < donchian_lower_aligned[i] or 
                ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian upper or weekly trend turns up
            if (price > donchian_upper_aligned[i] or 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolExp_WeeklyEMA50_v1"
timeframe = "4h"
leverage = 1.0