#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Champ_Volume_Signal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend and structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Daily volume confirmation - 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA21 + volatility contraction + volume expansion
            if (close[i] > ema_21_1w_aligned[i] and
                atr_1w_aligned[i] < atr_1w_aligned[i-1] * 0.9 and  # Volatility contraction
                vol_ratio[i] > 1.8):  # Volume expansion
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA21 + volatility contraction + volume expansion
            elif (close[i] < ema_21_1w_aligned[i] and
                  atr_1w_aligned[i] < atr_1w_aligned[i-1] * 0.9 and  # Volatility contraction
                  vol_ratio[i] > 1.8):  # Volume expansion
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: volatility expansion or price below weekly EMA21
            if (atr_1w_aligned[i] > atr_1w_aligned[i-1] * 1.1 or  # Volatility expansion
                close[i] < ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: volatility expansion or price above weekly EMA21
            if (atr_1w_aligned[i] > atr_1w_aligned[i-1] * 1.1 or  # Volatility expansion
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals