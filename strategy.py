#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_Trend_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Pad ER array
    er_full = np.zeros(n)
    er_full[er_period:] = er
    
    # Smoothing constants
    sc = (er_full * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Trend confirmation: 200-period SMA
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume filter: current volume > 1.5 * 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for SMA calculation
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(sma_200[i]) or np.isnan(vol_ma_50[i]):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        sma_val = sma_200[i]
        close_val = close[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: KAMA above SMA200 with volume
            if kama_val > sma_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA below SMA200 with volume
            elif kama_val < sma_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA crosses below SMA200
            if kama_val < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA crosses above SMA200
            if kama_val > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals