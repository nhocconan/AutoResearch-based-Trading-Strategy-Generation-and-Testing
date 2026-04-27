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
    
    # Get daily data for ATR and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate daily ATR ratio (current ATR / 20-day average ATR) for volatility expansion
    atr_ma_20 = np.zeros(len(df_1d))
    for i in range(20, len(atr_1d)):
        atr_ma_20[i] = np.mean(atr_1d[i-20:i])
    
    atr_ratio = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if atr_ma_20[i] > 0:
            atr_ratio[i] = atr_1d[i] / atr_ma_20[i]
        else:
            atr_ratio[i] = 1.0
    
    # Align ATR ratio to 6h timeframe (volatility expansion signal)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 60-minute data for trend filter
    df_60m = get_htf_data(prices, '60m')
    if len(df_60m) < 50:
        return np.zeros(n)
    
    close_60m = df_60m['close'].values
    
    # Calculate EMA(50) on 60-minute close for trend filter
    ema_60m_50 = np.zeros(len(df_60m))
    alpha = 2 / (50 + 1)
    for i in range(len(close_60m)):
        if i == 0:
            ema_60m_50[i] = close_60m[i]
        else:
            ema_60m_50[i] = close_60m[i] * alpha + ema_60m_50[i-1] * (1 - alpha)
    
    ema_60m_50_aligned = align_htf_to_ltf(prices, df_60m, ema_60m_50)
    
    # Calculate 6h ATR(14) for position sizing adjustment
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_6h = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_6h[i] = np.mean(tr[:i+1]) if i > 0 else tr[i]
        else:
            atr_6h[i] = (atr_6h[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50)  # volume MA needs 20, 60m EMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr_ratio_aligned[i]) or
            np.isnan(ema_60m_50_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        vol_expansion = atr_ratio_aligned[i] > 1.5  # Volatility expansion threshold
        trend_up = ema_60m_50_aligned[i] > ema_60m_50_aligned[i-1]
        trend_down = ema_60m_50_aligned[i] < ema_60m_50_aligned[i-1]
        
        if position == 0:
            # Long: volatility expansion + uptrend + volume confirmation
            if (vol_expansion and trend_up and vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: volatility expansion + downtrend + volume confirmation
            elif (vol_expansion and trend_down and vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: volatility contraction or trend reversal
            if (not vol_expansion or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: volatility contraction or trend reversal
            if (not vol_expansion or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_VolatilityExpansion_60mEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0