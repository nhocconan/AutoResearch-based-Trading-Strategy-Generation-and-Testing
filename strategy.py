#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing a reliable trend filter. 
Long when price > KAMA(10) and 1d EMA34 > prior 1d EMA34; short when price < KAMA(10) and 1d EMA34 < prior 1d EMA34. 
Volume confirmation (>1.5x 20-period average) reduces false signals. 
Designed for 4-hour timeframe with ~25-40 trades/year to minimize fee drag and work in both bull and bear markets via adaptive trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily typical price for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLC for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA(10) - Adaptive Moving Average
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smooth ER
    er_smooth = pd.Series(er).ewm(alpha=2/(10+1), adjust=False).fillna(0).values
    # Smoothing constants
    sc = (er_smooth * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day EMA trend filter (34-period)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_prev = np.roll(ema_1d, 1)
    ema_1d_prev[0] = ema_1d[0]
    ema_1d_rising = ema_1d > ema_1d_prev
    ema_1d_falling = ema_1d < ema_1d_prev
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_rising)
    ema_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_falling)
    
    # 4h volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_1d_rising_aligned[i]) or 
            np.isnan(ema_1d_falling_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        ema_rising = ema_1d_rising_aligned[i]
        ema_falling = ema_1d_falling_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA with rising 1d trend and volume
            if price > kama_val and ema_rising and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with falling 1d trend and volume
            elif price < kama_val and ema_falling and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below KAMA or trend turns down
            if price < kama_val or ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above KAMA or trend turns up
            if price > kama_val or ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_1d_Trend_Filter"
timeframe = "4h"
leverage = 1.0