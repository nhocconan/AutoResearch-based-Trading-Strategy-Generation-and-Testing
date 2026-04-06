#!/usr/bin/env python3
"""
4H Donchian 20 Breakout with Volume Confirmation and RSI Momentum Filter
Hypothesis: Donchian channel breakouts capture strong directional moves. Volume confirmation
ensures breakout strength, while RSI momentum filter avoids chasing overextended moves. 
Designed for 80-180 trades over 4 years (20-45/year) to balance opportunity with fee efficiency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # RSI calculation on 1d
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_rsi = 14
    avg_gain = wilder_smooth(gain, period_rsi)
    avg_loss = wilder_smooth(loss, period_rsi)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period MA)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20 + period_rsi, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or RSI extreme
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR RSI > 70 (overbought)
            if close[i] < donchian_low[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR RSI < 30 (oversold)
            if close[i] > donchian_high[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + RSI momentum
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            # RSI filter: avoid extremes, look for momentum
            rsi_momentum = (rsi_aligned[i] > 50 and rsi_aligned[i] < 70) or \
                           (rsi_aligned[i] < 50 and rsi_aligned[i] > 30)
            
            if bull_breakout and volume_filter and rsi_momentum:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and rsi_momentum:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals