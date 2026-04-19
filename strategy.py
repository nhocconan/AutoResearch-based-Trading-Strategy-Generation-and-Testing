#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_OverboughtOversold_VolumeATR_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI(14)
    delta = np.diff(df_1d['close'].values)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = gain_ma / loss_ma
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Daily ATR(14) for volatility
    tr = np.maximum(df_1d['high'].values[1:] - df_1d['low'].values[1:], 
                    np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr = np.maximum(tr, np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_14_aligned[i]
        atr = atr_14_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: RSI oversold (<30) with volume
            if rsi < 30 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with volume
            elif rsi > 70 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (40-60) or ATR stop
            if rsi > 40 or price < close[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (40-60) or ATR stop
            if rsi < 60 or price > close[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals