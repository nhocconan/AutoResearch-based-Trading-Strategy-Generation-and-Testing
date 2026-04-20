#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h and 1d data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # 12h volume ratio (current / 20-period average)
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / np.where(vol_ma_20_12h == 0, 1, vol_ma_20_12h)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # 1d close for trend
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 6h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_14_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_14_12h_aligned[i]
        vol_ratio_12h = vol_ratio_12h_aligned[i]
        sma_50_1d = sma_50_1d_aligned[i]
        vol_ratio_6h = vol_ratio[i]
        
        # Volatility filter: avoid extremes
        atr_ma_20 = pd.Series(atr_14_12h_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume on both timeframes
        vol_filter = vol_filter and (vol_ratio_6h > 1.5) and (vol_ratio_12h > 1.3)
        
        # Trend filter: price above/below 1d SMA50
        trend_up = price > sma_50_1d
        trend_down = price < sma_50_1d
        
        if position == 0:
            # Enter long in uptrend with volume and volatility filter
            if trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend with volume and volatility filter
            elif trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend breakdown or volatility spike
            if (not trend_up) or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend breakdown or volatility spike
            if (not trend_down) or (atr > 3.5 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_1d_ATR_Volume_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0