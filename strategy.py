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
    
    # Load 1d data for volatility and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility
    atr_14_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        tr = np.maximum(high_1d[1:] - low_1d[1:], 
                        np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                   np.abs(low_1d[1:] - close_1d[:-1])))
        tr = np.concatenate([[np.nan], tr])
        atr_14_1d[13] = np.nanmean(tr[1:15])
        for i in range(14, len(close_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily EMA50 for trend
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Calculate 12h Bollinger Bands
    bb_length = 20
    bb_mult = 2.0
    bb_basis = np.full(n, np.nan)
    bb_dev = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    if n >= bb_length:
        for i in range(bb_length-1, n):
            bb_basis[i] = np.mean(close[i-bb_length+1:i+1])
            bb_dev[i] = bb_mult * np.std(close[i-bb_length+1:i+1])
            bb_upper[i] = bb_basis[i] + bb_dev[i]
            bb_lower[i] = bb_basis[i] - bb_dev[i]
    
    # Align 1d indicators to 12h
    atr_14_1d_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_1d_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Band width % (volatility normalized)
    bb_width_pct = np.full(n, np.nan)
    if n >= bb_length:
        for i in range(bb_length-1, n):
            if bb_basis[i] != 0:
                bb_width_pct[i] = (bb_upper[i] - bb_lower[i]) / bb_basis[i] * 100
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_1d_12h[i]) or 
            np.isnan(ema_50_1d_12h[i]) or
            np.isnan(bb_width_pct[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: only trade when volatility is elevated
        # Using 50-period average of BB width to determine high volatility regime
        if i >= 150:  # Need enough history for BB width average
            bb_width_ma = np.mean(bb_width_pct[i-49:i+1])  # 50-period MA of BB width
            if bb_width_ma < 2.0:  # Low volatility - avoid trading
                signals[i] = 0.0
                continue
        
        if position == 0:
            # Long: Price touches lower BB with high volatility and above daily EMA50
            if (close[i] <= bb_lower[i] and
                close[i] > ema_50_1d_12h[i]):
                position = 1
                signals[i] = position_size
            # Short: Price touches upper BB with high volatility and below daily EMA50
            elif (close[i] >= bb_upper[i] and
                  close[i] < ema_50_1d_12h[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price touches middle BB or crosses below EMA50
            if (close[i] >= bb_basis[i] or 
                close[i] < ema_50_1d_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price touches middle BB or crosses above EMA50
            if (close[i] <= bb_basis[i] or 
                close[i] > ema_50_1d_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_BB_Touch_EMA50_Volatility_Filter"
timeframe = "12h"
leverage = 1.0