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
    
    # Daily data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily volatility ratio (current ATR / 20-period average ATR)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = atr_1d / atr_ma_20_1d
    
    # Align data to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: trade only when volatility is elevated (above average)
        vol_condition = vol_ratio_aligned[i] > 1.2
        
        # Trend filter: price above/below daily EMA20
        long_trend = close[i] > ema_20_1d_aligned[i]
        short_trend = close[i] < ema_20_1d_aligned[i]
        
        # Entry conditions: 
        # Long when price is above EMA20 in high volatility environment
        # Short when price is below EMA20 in high volatility environment
        if position == 0:
            if long_trend and vol_condition:
                position = 1
                signals[i] = position_size
            elif short_trend and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price crosses below EMA20 or volatility drops
            if not long_trend or vol_ratio_aligned[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price crosses above EMA20 or volatility drops
            if not short_trend or vol_ratio_aligned[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA20_Volatility_Filter"
timeframe = "4h"
leverage = 1.0