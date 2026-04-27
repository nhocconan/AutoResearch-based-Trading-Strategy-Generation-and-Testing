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
    
    # Get daily data for OHLC and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if i == 14:
            atr_1d[i] = np.mean(tr_1d[:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Daily ATR ratio: current ATR / 20-period ATR average (volatility regime)
    atr_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(atr_1d)):
        atr_ma_20[i] = np.mean(atr_1d[i-20:i])
    
    atr_ratio = np.full(len(df_1d), np.nan)
    for i in range(20, len(atr_1d)):
        if atr_ma_20[i] > 0:
            atr_ratio[i] = atr_1d[i] / atr_ma_20[i]
    
    # Align ATR ratio to 1h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 4h close for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend direction
    ema_4h_50 = np.full(len(df_4h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_4h)):
        if i < 49:
            ema_4h_50[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_4h_50[i-1]):
                ema_4h_50[i] = np.mean(close_4h[i-49:i+1])
            else:
                ema_4h_50[i] = close_4h[i] * alpha + ema_4h_50[i-1] * (1 - alpha)
    
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Calculate 1h ATR(14) for position sizing volatility adjustment
    tr1_h = high[1:] - low[1:]
    tr2_h = np.abs(high[1:] - close[:-1])
    tr3_h = np.abs(low[1:] - close[:-1])
    tr_h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))])
    
    atr_1h = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr_1h[i] = np.mean(tr_h[:15])
        else:
            atr_1h[i] = (atr_1h[i-1] * 13 + tr_h[i]) / 14
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need 4h EMA(50) and daily ATR ratio
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_4h_50_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR ratio > 1.2)
        vol_filter = atr_ratio_aligned[i] > 1.2
        
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above rising 4h EMA50 with volatility expansion
            if (price > ema_4h_50_aligned[i] and 
                ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: price below falling 4h EMA50 with volatility expansion
            elif (price < ema_4h_50_aligned[i] and 
                  ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA50 or volatility contraction
            if (price < ema_4h_50_aligned[i] or 
                atr_ratio_aligned[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above EMA50 or volatility contraction
            if (price > ema_4h_50_aligned[i] or 
                atr_ratio_aligned[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolatilityFilter_4hEMA50_Trend_v1"
timeframe = "1h"
leverage = 1.0