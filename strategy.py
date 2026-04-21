#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_1dEMA50
Hypothesis: Donchian(20) breakouts in the direction of the 1d EMA50 trend with volume confirmation capture medium-term trends in both bull and bear markets. The 1d EMA50 provides a robust trend filter, while volume confirmation ensures breakouts are genuine. This strategy targets 20-40 trades per year on 4h, minimizing fee drag. Uses ATR-based stop loss via signal=0.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Main timeframe data (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            highest_high[i] = np.max(high[max(0, i-19):i+1]) if i >= 0 else high[i]
            lowest_low[i] = np.min(low[max(0, i-19):i+1]) if i >= 0 else low[i]
        else:
            highest_high[i] = np.max(high[i-20:i+1])
            lowest_low[i] = np.min(low[i-20:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average (volume confirmation)
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(25, n):  # Start after Donchian warmup
        # Skip if NaN in critical values
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        highest = highest_high[i]
        lowest = lowest_low[i]
        ema50 = ema50_1d_aligned[i]
        vol_confirm = volume_confirm[i]
        
        # Stoploss: 2.5 * ATR from entry (using 14-period ATR approximation)
        # Calculate ATR on the fly for simplicity in stop condition
        if i >= 14:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            # Simplified ATR: use recent true range average
            tr_sum = 0
            for j in range(1, 15):
                if i - j >= 0:
                    tr_sum += max(high[i-j] - low[i-j], abs(high[i-j] - close[i-j-1]), abs(low[i-j] - close[i-j-1]))
            atr_est = tr_sum / 14
        else:
            atr_est = 0
        
        if position == 1 and price < entry_price - 2.5 * atr_est:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_est:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume confirmation
            if price > highest and price > ema50 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low in downtrend with volume confirmation
            elif price < lowest and price < ema50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to Donchian mid or trend breaks
            mid = (highest + lowest) / 2.0
            if price < mid or price < ema50:  # Return to mid or trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian mid or trend breaks
            mid = (highest + lowest) / 2.0
            if price > mid or price > ema50:  # Return to mid or trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_Trend_1dEMA50"
timeframe = "4h"
leverage = 1.0