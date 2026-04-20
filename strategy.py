#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    
    # Daily SMA(50) for intermediate trend
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Daily SMA(200) for long-term trend
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume filter (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(sma_50_1d_aligned[i]) or np.isnan(sma_200_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        sma_50 = sma_50_1d_aligned[i]
        sma_200 = sma_200_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_4h = vol_ratio[i]
        
        # Multi-timeframe trend alignment: price above both SMAs for uptrend
        trend_up = (price > sma_50) and (sma_50 > sma_200)
        # Price below both SMAs for downtrend
        trend_down = (price < sma_50) and (sma_50 < sma_200)
        
        # Volatility filter: avoid low volatility (chop) and extreme volatility
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_4h > 1.5)
        
        if position == 0:
            # Enter long in strong uptrend with volume
            if trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in strong downtrend with volume
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

name = "4h_1d_SMA50_200_Trend_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0