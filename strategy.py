#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_200ema_plus_200sma_trend_follow_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily EMA200 and SMA200 ===
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    sma200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Align to 4h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    sma200_aligned = align_htf_to_ltf(prices, df_1d, sma200)
    
    # === 4h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema200_aligned[i]
        sma_val = sma200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(sma_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above both EMA200 and SMA200 with volume confirmation
            if close_val > ema_val and close_val > sma_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price below both EMA200 and SMA200 with volume confirmation
            elif close_val < ema_val and close_val < sma_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below EMA200 OR SMA200
            if close_val < ema_val or close_val < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above EMA200 OR SMA200
            if close_val > ema_val or close_val > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals