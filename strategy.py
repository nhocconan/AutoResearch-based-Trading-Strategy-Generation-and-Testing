#!/usr/bin/env python3
"""
4h_ADX_Trend_Strength_With_Volume_Confirmation
Hypothesis: Strong ADX trend (ADX>25) combined with volume spikes (1.5x 20-period average) 
provides high-probability entries in the direction of the 4h EMA(50) trend. 
Volume confirms institutional participation, reducing false breakouts. 
Designed for 4h timeframe to capture multi-day trends with low frequency 
(20-40 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * np.arange(1, n+1) + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * np.arange(1, n+1) + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # EMA(50) trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        ema_val = ema_50[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: ADX > 25 (strong trend), price above EMA50, volume confirmation
            if adx_val > 25 and price > ema_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend), price below EMA50, volume confirmation
            elif adx_val > 25 and price < ema_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if trend weakens (ADX < 20) or price crosses below EMA50
            if adx[i] < 20 or price < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if trend weakens (ADX < 20) or price crosses above EMA50
            if adx[i] < 20 or price > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_Trend_Strength_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0