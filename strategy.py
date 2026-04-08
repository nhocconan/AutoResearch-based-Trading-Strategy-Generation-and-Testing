#!/usr/bin/env python3
# 4h_rsi_adaptive_volume_filter_v1
# Hypothesis: RSI-based mean reversion with volume confirmation and volatility filter. 
# Uses RSI(14) extremes (30/70) for entries, volume > 1.5x average for confirmation, 
# and ATR-based volatility filter to avoid choppy markets. Designed for low trade frequency
# (target: 20-30 trades/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_adaptive_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or np.isnan(atr[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop) and extreme volatility
        atr_ratio = atr[i] / np.mean(atr[max(0, i-50):i+1]) if i >= 50 else 1.0
        vol_filter = 0.5 <= atr_ratio <= 2.0  # Trade only in moderate volatility regimes
        
        if position == 1:  # Long position
            # Exit on RSI overbought or volatility breakdown
            if rsi[i] > 70 or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on RSI oversold or volatility breakdown
            if rsi[i] < 30 or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Mean reversion entries with volume and volatility filter
            if rsi[i] < 30 and volume_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            elif rsi[i] > 70 and volume_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals