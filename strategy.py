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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 22-period ATR for volatility filter (11 days at 2h intervals)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    for i in range(22, len(tr)):
        if i == 22:
            atr[i] = np.mean(tr[1:23])
        else:
            atr[i] = (atr[i-1] * 21 + tr[i]) / 22
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 22, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price above weekly EMA50 with volume confirmation
            if price > ema_1w_aligned[i] and vol_ratio > 2.5:
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA50 with volume confirmation
            elif price < ema_1w_aligned[i] and vol_ratio > 2.5:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly EMA50
            if price < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly EMA50
            if price > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA50_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0