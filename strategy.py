#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 1d volume moving average (20-day)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h SMA(50) for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(sma_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_val = atr_1d_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price above SMA50 with volatility expansion and volume surge
            if price > sma_50[i] and atr_val > 1.5 * atr_1d_aligned[i-1] and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below SMA50 with volatility expansion and volume surge
            elif price < sma_50[i] and atr_val > 1.5 * atr_1d_aligned[i-1] and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: volatility contraction or price crosses below SMA50
            if atr_val < 0.8 * atr_1d_aligned[i-1] or price < sma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: volatility contraction or price crosses above SMA50
            if atr_val < 0.8 * atr_1d_aligned[i-1] or price > sma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolatilityExpansion_VolumeSurge_SMA50Filter"
timeframe = "4h"
leverage = 1.0