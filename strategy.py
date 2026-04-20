#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h ATR for volatility filter
    high_low_4h = high_4h - low_4h
    high_close_4h = np.abs(high_4h - np.roll(close_4h, 1))
    low_close_4h = np.abs(low_4h - np.roll(close_4h, 1))
    high_low_4h[0] = high_4h[0] - low_4h[0]
    high_close_4h[0] = np.abs(high_4h[0] - close_4h[0])
    low_close_4h[0] = np.abs(low_4h[0] - close_4h[0])
    tr_4h = np.maximum(high_low_4h, np.maximum(high_close_4h, low_close_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume for confirmation
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(vol_ma_4h[i]) or np.isnan(close_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        atr = atr_4h[i]
        vol_ma = vol_ma_4h[i]
        
        if position == 0:
            # Long: price above daily EMA50 with volume expansion and sufficient volatility
            if (price > ema_50_1d_aligned[i] and 
                vol > 1.5 * vol_ma and 
                atr > 0):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA50 with volume expansion and sufficient volatility
            elif (price < ema_50_1d_aligned[i] and 
                  vol > 1.5 * vol_ma and 
                  atr > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA50 or volatility drops
            if price < ema_50_1d_aligned[i] or vol < 0.5 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA50 or volatility drops
            if price > ema_50_1d_aligned[i] or vol < 0.5 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyEMA50_VolumeExpansion_V1"
timeframe = "4h"
leverage = 1.0