#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and volatility
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA50 on weekly close for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 14-period ATR on weekly for volatility filter
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(tr_1w), np.nan)
    for i in range(14, len(tr_1w)):
        if i == 14:
            atr_1w[i] = np.mean(tr_1w[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Align weekly ATR to daily
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 20-period volume average on weekly
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    vol_period = 20
    for i in range(vol_period, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-vol_period:i])
    
    # Align weekly volume MA to daily
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1w_aligned[i] if vol_ma_1w_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: Price above weekly EMA50 with volume confirmation and volatility filter
            if price > ema_1w_aligned[i] and vol_ratio > 1.5 and atr_1w_aligned[i] > 0:
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA50 with volume confirmation and volatility filter
            elif price < ema_1w_aligned[i] and vol_ratio > 1.5 and atr_1w_aligned[i] > 0:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly EMA50 or volatility drops
            if price < ema_1w_aligned[i] or atr_1w_aligned[i] < 0.5 * np.mean(atr_1w_aligned[max(0, i-20):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly EMA50 or volatility drops
            if price > ema_1w_aligned[i] or atr_1w_aligned[i] < 0.5 * np.mean(atr_1w_aligned[max(0, i-20):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA50_WeeklyTrend_Volume_Filter"
timeframe = "1d"
leverage = 1.0