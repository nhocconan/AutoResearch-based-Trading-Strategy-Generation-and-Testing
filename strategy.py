#!/usr/bin/env python3
name = "4h_Trix_Volume_Spike_Regime"
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
    
    # Get 4h data for TRIX calculation (HTF for signal generation)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA applied 3 times)
    close_4h = df_4h['close'].values
    ema1 = pd.Series(close_4h).ewm(span=15, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Align TRIX to 4h timeframe
    trix = align_htf_to_ltf(prices, df_4h, trix_raw)
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Chop regime filter: avoid trending markets
    # Calculate EMA50 and EMA200 for trend strength
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
    # Distance from EMA200 normalized by ATR-like measure
    price_position = (close - ema200) / (ema50 - ema200 + 1e-10)
    # Chop filter: avoid when too far from EMA200 (strong trend)
    chop_filter = np.abs(price_position) < 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero AND above 12h EMA50 (uptrend) AND volume spike AND not in strong trend
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_12h_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND below 12h EMA50 (downtrend) AND volume spike AND not in strong trend
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_12h_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR below 12h EMA50 (trend change)
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: TRIX crosses above zero OR above 12h EMA50 (trend change)
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals