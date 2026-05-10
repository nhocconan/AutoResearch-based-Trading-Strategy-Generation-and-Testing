#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_12hEMA50_Volume
# Hypothesis: Combines Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian high(20) with volume > 1.5x average and price > 12h EMA50.
# Short when price breaks below Donchian low(20) with volume > 1.5x average and price < 12h EMA50.
# Exits when price crosses 12h EMA50 in opposite direction.
# Designed for 30-50 trades/year to avoid overtrading and work in both bull and bear markets.

name = "4h_Donchian_Breakout_20_12hEMA50_Volume"
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
    
    # Donchian Channel (20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume average (20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high with volume confirmation and uptrend
            if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low with volume confirmation and downtrend
            elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below 12h EMA50
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above 12h EMA50
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals