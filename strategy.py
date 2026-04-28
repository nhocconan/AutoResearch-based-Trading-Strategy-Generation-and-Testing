#!/usr/bin/env python3
"""
1h_Telegraph_Signal_v1
Hypothesis: Uses 4h Supertrend for trend direction, 1h RSI(14) with 30/70 levels for entry timing, and volume confirmation (1.5x 20-bar average) to capture high-probability pullback entries in trending markets. Designed for low trade frequency (15-37/year) to minimize fee drag while capturing swings in both bull and bear markets. The Supertrend filter ensures we only trade with the higher timeframe trend, reducing whipsaws. RSI provides mean-reversion entries within the trend, and volume confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on 4h: ATR(10), multiplier=3.0
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([ [np.nan], np.maximum(tr1, np.maximum(tr2, tr3)) ])
    
    # ATR(10)
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_4h + low_4h) / 2
    upper_basic = hl2 + (3.0 * atr)
    lower_basic = hl2 - (3.0 * atr)
    
    # Final Upper and Lower Bands
    upper_final = np.full_like(close_4h, np.nan)
    lower_final = np.full_like(close_4h, np.nan)
    upper_final[0] = upper_basic[0]
    lower_final[0] = lower_basic[0]
    
    for i in range(1, len(close_4h)):
        if upper_basic[i] < upper_final[i-1] or close_4h[i-1] > upper_final[i-1]:
            upper_final[i] = upper_basic[i]
        else:
            upper_final[i] = upper_final[i-1]
            
        if lower_basic[i] > lower_final[i-1] or close_4h[i-1] < lower_final[i-1]:
            lower_final[i] = lower_basic[i]
        else:
            lower_final[i] = lower_final[i-1]
    
    # Supertrend direction
    supertrend = np.full_like(close_4h, np.nan)
    supertrend[0] = upper_final[0]
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_final[i-1]:
            supertrend[i] = lower_final[i]
        elif close_4h[i] < lower_final[i-1]:
            supertrend[i] = upper_final[i]
        else:
            supertrend[i] = supertrend[i-1]
    
    # Trend: 1 if price > Supertrend (uptrend), -1 if price < Supertrend (downtrend)
    trend_dir = np.where(close_4h > supertrend, 1, -1)
    trend_dir = np.where(np.isnan(trend_dir), 0, trend_dir)
    
    # Align trend direction to 1h
    trend_dir_aligned = align_htf_to_ltf(prices, df_4h, trend_dir)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 10)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_dir_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        trend = int(trend_dir_aligned[i])
        rsi_val = rsi[i]
        vol_ok = vol_confirm[i]
        
        # Entry conditions: RSI pullback in trend direction with volume
        long_entry = (trend == 1) and (rsi_val < 30) and vol_ok
        short_entry = (trend == -1) and (rsi_val > 70) and vol_ok
        
        # Exit conditions: RSI reverts to midpoint (50) or trend change
        long_exit = (rsi_val >= 50) or (trend != 1)
        short_exit = (rsi_val <= 50) or (trend != -1)
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Telegraph_Signal_v1"
timeframe = "1h"
leverage = 1.0