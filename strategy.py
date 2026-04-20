#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 14-day ATR for volatility measurement
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 20-day SMA for trend filter
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # 50-day SMA for longer-term trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # Volume confirmation: current volume vs 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(sma_20_aligned[i]) or np.isnan(sma_50_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price above both SMAs with volume confirmation and sufficient volatility
            if (price > sma_20_aligned[i] and price > sma_50_aligned[i] and 
                vol > 1.5 * vol_ma_20_aligned[i] and 
                atr_14_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price below both SMAs with volume confirmation and sufficient volatility
            elif (price < sma_20_aligned[i] and price < sma_50_aligned[i] and 
                  vol > 1.5 * vol_ma_20_aligned[i] and 
                  atr_14_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day SMA or volatility drops significantly
            if price < sma_20_aligned[i] or vol < 0.6 * vol_ma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day SMA or volatility drops significantly
            if price > sma_20_aligned[i] or vol < 0.6 * vol_ma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DualSMA25_VolumeFilter"
timeframe = "1d"
leverage = 1.0