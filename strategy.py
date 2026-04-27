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
    
    # Get daily data for ATR and Bollinger Bands (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for Donchian channel (price channel) - primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) channel
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1h data for momentum filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # 1h RSI(14) for momentum
    close_1h = df_1h['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(rsi_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr_val = atr_1d_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        rsi_val = rsi_1h_aligned[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        atr_ma = pd.Series(atr_1d_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1] if i >= 20 else atr_val
        vol_filter = atr_val > atr_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian with RSI > 50 and high volatility
            if close[i] > upper_val and rsi_val > 50 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with RSI < 50 and high volatility
            elif close[i] < lower_val and rsi_val < 50 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian or RSI < 30
            if close[i] < lower_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian or RSI > 70
            if close[i] > upper_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_RSI14_ATR_Vol_Filter"
timeframe = "4h"
leverage = 1.0