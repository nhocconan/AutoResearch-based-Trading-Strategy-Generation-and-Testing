#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for structure and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Weekly close for trend filter
    weekly_close = close_1w
    
    # 6h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h ATR(14) for volatility
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = tr1_6h[0]
    tr3_6h[0] = tr1_6h[0]
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume ratio (current / 50-period average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma_50 == 0, 1, vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_14_1w_aligned[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_close_val = weekly_close[i]
        atr_1w = atr_14_1w_aligned[i]
        atr_6h = atr_14_6h[i]
        vol_ratio_6h = vol_ratio[i]
        
        # Weekly trend: price above/below weekly close
        weekly_uptrend = price > weekly_close_val
        weekly_downtrend = price < weekly_close_val
        
        # Volatility filter: avoid low volatility (chop) and extreme volatility
        atr_ratio = atr_6h / atr_1w if atr_1w > 0 else 0
        vol_filter = (atr_ratio > 0.3) and (atr_ratio < 3.0)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_6h > 1.5)
        
        if position == 0:
            # Enter long in weekly uptrend with volume and volatility filter
            if weekly_uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in weekly downtrend with volume and volatility filter
            elif weekly_downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend breakdown or volatility spike
            if (not weekly_uptrend) or (atr_ratio > 4.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend breakdown or volatility spike
            if (not weekly_downtrend) or (atr_ratio > 4.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_ATR_Volume_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0