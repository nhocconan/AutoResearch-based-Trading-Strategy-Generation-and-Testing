#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend and volatility (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily 200 EMA for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily ATR for volatility filter and stop
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume average for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h high/low for breakout levels (using 4h data)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4-period high/low for breakout
    high_4_4h = pd.Series(high_4h).rolling(window=4, min_periods=4).max().values
    low_4_4h = pd.Series(low_4h).rolling(window=4, min_periods=4).min().values
    high_4_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4_4h)
    low_4_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(high_4_4h_aligned[i]) or 
            np.isnan(low_4_4h_aligned[i]) or np.isnan(close_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_1d[i]  # Use daily volume for confirmation
        
        if position == 0:
            # Long: price breaks above 4-period high with volume confirmation and above daily EMA200
            if (price > high_4_4h_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                price > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4-period low with volume confirmation and below daily EMA200
            elif (price < low_4_4h_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  price < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4-period low or volume drops significantly
            if price < low_4_4h_aligned[i] or vol < 0.8 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 4-period high or volume drops significantly
            if price > high_4_4h_aligned[i] or vol < 0.8 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Breakout4_Volume_EMA200Filter"
timeframe = "4h"
leverage = 1.0