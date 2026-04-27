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
    
    # Get daily data for ATR and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(high_1d), np.nan)
    for i in range(14, len(tr_1d)):
        if i == 14:
            atr_14_1d[i] = np.mean(tr_1d[:i+1])
        else:
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate daily CCI(20)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    sma_tp_20 = np.full(len(typical_price_1d), np.nan)
    for i in range(19, len(typical_price_1d)):
        sma_tp_20[i] = np.mean(typical_price_1d[i-19:i+1])
    
    mad_20 = np.full(len(typical_price_1d), np.nan)
    for i in range(19, len(typical_price_1d)):
        dev = np.abs(typical_price_1d[i-19:i+1] - sma_tp_20[i])
        mad_20[i] = np.mean(dev)
    
    cci_20_1d = np.full(len(typical_price_1d), np.nan)
    for i in range(19, len(typical_price_1d)):
        if mad_20[i] > 0:
            cci_20_1d[i] = (typical_price_1d[i] - sma_tp_20[i]) / (0.015 * mad_20[i])
    
    # Align daily indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    cci_20_aligned = align_htf_to_ltf(prices, df_1d, cci_20_1d)
    
    # Calculate 4h ATR(14) for position sizing and stop
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    atr_14_4h = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr_14_4h[i] = np.mean(tr_4h[:i+1])
        else:
            atr_14_4h[i] = (atr_14_4h[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all indicators
    start_idx = max(20, 19)  # volume MA needs 20, daily indicators need 19
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_aligned[i]) or
            np.isnan(cci_20_aligned[i]) or
            np.isnan(atr_14_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # ATR-based dynamic threshold
        atr_threshold = atr_14_aligned[i] * 0.5
        
        if position == 0:
            # Long: CCI < -100 (oversold) + price near daily low + volume
            if (cci_20_aligned[i] < -100 and 
                price <= low_1d[np.searchsorted(df_1d.index.values[:i+1], prices.index[i], side='right') - 1] + atr_threshold and
                volume_confirmation):
                signals[i] = 0.25
                position = 1
            # Short: CCI > 100 (overbought) + price near daily high + volume
            elif (cci_20_aligned[i] > 100 and 
                  price >= high_1d[np.searchsorted(df_1d.index.values[:i+1], prices.index[i], side='right') - 1] - atr_threshold and
                  volume_confirmation):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: CCI > -20 (recovering from oversold) or ATR-based stop
            if (cci_20_aligned[i] > -20 or 
                price <= high_1d[np.searchsorted(df_1d.index.values[:i+1], prices.index[i], side='right') - 1] - 2.0 * atr_14_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI < 20 (declining from overbought) or ATR-based stop
            if (cci_20_aligned[i] < 20 or 
                price >= low_1d[np.searchsorted(df_1d.index.values[:i+1], prices.index[i], side='right') - 1] + 2.0 * atr_14_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyCCI_ATR_Volume_Reversion_v1"
timeframe = "4h"
leverage = 1.0