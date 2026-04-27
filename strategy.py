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
    
    # Get daily data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * multiplier) + (ema_34_1d[i-1] * (1 - multiplier))
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = np.nan
    
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - close_1d_prev),
                               np.abs(low_1d - close_1d_prev)))
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.nanmean(tr[:14])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (tr[i] * (13/14)) + (atr_14_1d[i-1] * (1/14))
    
    # Align daily indicators to 1d timeframe (no additional shift needed)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        
        # Volume spike filter: at least 2x average volume
        vol_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: Price above EMA34 with volume spike and momentum
            if price > ema_34_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA34 with volume spike and momentum
            elif price < ema_34_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA34 or volatility collapse
            if price < ema_34_1d_aligned[i] or (vol_ratio < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA34 or volatility collapse
            if price > ema_34_1d_aligned[i] or (vol_ratio < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0