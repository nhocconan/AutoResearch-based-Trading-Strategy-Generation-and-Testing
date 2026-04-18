#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1d_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend signals on 12h timeframe.
Combined with 1d EMA34 trend filter and volume confirmation (volume > 1.5x 20-period average),
this strategy captures strong trends while avoiding whipsaws in ranging markets.
Designed for low trade frequency (15-30 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Handle volatility calculation properly
    volatility = np.convolve(np.abs(np.diff(close)), np.ones(10), mode='same')
    volatility[:9] = np.nan  # Not enough data for full window
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods for volatility calculation
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Warmup for volume MA and KAMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema_trend = ema_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA and above 1d EMA with volume
            if price > kama_val and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below 1d EMA with volume
            elif price < kama_val and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price crosses below KAMA or trend turns bearish
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above KAMA or trend turns bullish
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0