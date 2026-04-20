#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and momentum filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-day high/low) for breakout
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h (previous day's Donchian levels available at 6h open)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # 1d RSI(14) for momentum filter
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.finfo(float).eps, avg_loss)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: break above 20-day high with volume and bullish momentum (RSI > 50)
            if price > high_20_aligned[i] and vol > 1.5 * vol_ma[i] and rsi_14_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low with volume and bearish momentum (RSI < 50)
            elif price < low_20_aligned[i] and vol > 1.5 * vol_ma[i] and rsi_14_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price retests the 20-day low or momentum fades
            if price < low_20_aligned[i] or rsi_14_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price retests the 20-day high or momentum fades
            if price > high_20_aligned[i] or rsi_14_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_RSI14_VolumeBreakout"
timeframe = "6h"
leverage = 1.0