#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Daily ATR for volatility
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Daily EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly EMA200 for long-term trend
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price above daily EMA50, above weekly EMA200, with volume and volatility
            if (price > ema_50_1d_aligned[i] and 
                price > ema_200_1w_aligned[i] and 
                vol > 1.3 * vol_ma_1d_aligned[i] and 
                atr_1d_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA50, below weekly EMA200, with volume and volatility
            elif (price < ema_50_1d_aligned[i] and 
                  price < ema_200_1w_aligned[i] and 
                  vol > 1.3 * vol_ma_1d_aligned[i] and 
                  atr_1d_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA50 or below weekly EMA200
            if price < ema_50_1d_aligned[i] or price < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA50 or above weekly EMA200
            if price > ema_50_1d_aligned[i] or price > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DualEMA_Filter_Volume_V1"
timeframe = "12h"
leverage = 1.0