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
    
    # Get 12h data for trend direction and volatility
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA200 for trend filter
    ema_period = 200
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate 12h ATR for volatility filter
    tr_12h = np.maximum(high_12h[1:] - low_12h[1:], 
                        np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                                   np.abs(low_12h[1:] - close_12h[:-1])))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_period = 14
    atr_12h = np.full(len(tr_12h), np.nan)
    for i in range(atr_period, len(tr_12h)):
        atr_12h[i] = np.mean(tr_12h[i-atr_period:i])
    
    # Get 1d data for volume profile and pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate 1d standard deviation for Bollinger-like bands
    vwap_dev = np.std(typical_price_1d[-20:]) if len(typical_price_1d) >= 20 else np.std(typical_price_1d)
    upper_vwap = vwap_1d + 2 * vwap_dev
    lower_vwap = vwap_1d - 2 * vwap_dev
    
    # Align indicators to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    upper_vwap_aligned = align_htf_to_ltf(prices, df_1d, upper_vwap)
    lower_vwap_aligned = align_htf_to_ltf(prices, df_1d, lower_vwap)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = max(ema_period, atr_period, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(upper_vwap_aligned[i]) or 
            np.isnan(lower_vwap_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        trend_up = price > ema_12h_aligned[i]
        volatility_normal = atr_12h_aligned[i] > 0  # Basic volatility check
        
        if position == 0:
            # Long: price near lower VWAP band + volume spike + uptrend
            if (price <= lower_vwap_aligned[i] * 1.02 and  # Near lower band
                vol_ratio > 1.5 and 
                trend_up):
                signals[i] = size
                position = 1
            # Short: price near upper VWAP band + volume spike + downtrend
            elif (price >= upper_vwap_aligned[i] * 0.98 and  # Near upper band
                  vol_ratio > 1.5 and 
                  not trend_up):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses VWAP or trend reversal
            if (price >= vwap_1d_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses VWAP or trend reversal
            if (price <= vwap_1d_aligned[i] or 
                trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_VWAP_Bands_Volume_Trend"
timeframe = "6h"
leverage = 1.0