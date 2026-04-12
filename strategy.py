#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_1d = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Align daily ATR to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = np.zeros_like(close_1w, dtype=float)
    ema_21_1w[:] = np.nan
    if len(close_1w) >= 21:
        multiplier = 2 / (21 + 1)
        ema_21_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = (close_1w[i] - ema_21_1w[i-1]) * multiplier + ema_21_1w[i-1]
    
    # Align weekly EMA to 4h timeframe
    ema_21_4h = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = high
    low_4h = low
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_4h[i]) or np.isnan(ema_21_4h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below EMA21
        price_above_ema = close[i] > ema_21_4h[i]
        price_below_ema = close[i] < ema_21_4h[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # ATR-based volatility filter: only trade when ATR is expanding
        atr_expanding = atr_4h[i] > np.mean(atr_4h[max(0, i-20):i]) if i >= 20 else False
        
        # Entry conditions
        if long_breakout and price_above_ema and atr_expanding and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and price_below_ema and atr_expanding and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite Donchian breakout or ATR contraction
        elif position == 1 and (close[i] < donchian_low[i] or not atr_expanding):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or not atr_expanding):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1w_donchian_atr_trend_filter_v1"
timeframe = "4h"
leverage = 1.0