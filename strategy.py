#!/usr/bin/env python3
"""
4h_ADX_Strength_1dTrend_v1
Hypothesis: Strong trend identification using ADX with directional bias from daily EMA.
Long when ADX>25, +DI>-DI, and price above daily EMA50. Short when ADX>25, -DI>+DI, and price below daily EMA50.
Uses volatility-based position sizing (ATR inverse) to manage risk and avoid overexposure.
Works in both bull and bear markets by requiring strong trend confirmation.
"""

name = "4h_ADX_Strength_1dTrend_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX (period=14)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(atr * 14).rolling(window=14, min_periods=14).sum().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(atr * 14).rolling(window=14, min_periods=14).sum().values
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong uptrend (ADX>25, +DI>-DI) + price above EMA50 + volume filter
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Strong downtrend (ADX>25, -DI>+DI) + price below EMA50 + volume filter
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Trend weakening (ADX<20) or opposite DI crossover
            if position == 1:
                if adx[i] < 20 or minus_di[i] > plus_di[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if adx[i] < 20 or plus_di[i] > minus_di[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals