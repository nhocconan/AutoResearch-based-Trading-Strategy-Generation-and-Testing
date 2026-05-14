#!/usr/bin/env python3
"""
4h_RSI_Momentum_Trend_Filter
Strategy: 4h RSI momentum with 1d trend filter.
Long: RSI > 50 and rising + price above 1d EMA34 + volume > 1.5x average
Short: RSI < 50 and falling + price below 1d EMA34 + volume > 1.5x average
Exit: RSI crosses back to 50
Position size: 0.25
Designed to capture momentum moves aligned with daily trend.
Timeframe: 4h
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
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI momentum: current RSI vs previous RSI
    rsi_momentum = rsi - np.roll(rsi, 1)
    rsi_momentum[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # RSI conditions
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        rsi_rising = rsi_momentum[i] > 0
        rsi_falling = rsi_momentum[i] < 0
        
        if position == 0:
            # Long: RSI > 50 and rising + price above EMA + volume filter
            if rsi_above_50 and rsi_rising and price_above_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 and falling + price below EMA + volume filter
            elif rsi_below_50 and rsi_falling and price_below_ema and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI falls back to 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI rises back to 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Momentum_Trend_Filter"
timeframe = "4h"
leverage = 1.0