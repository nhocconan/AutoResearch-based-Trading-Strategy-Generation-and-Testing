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
    
    # Get 1d data for ATR and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    atr_period = 14
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(tr), np.nan)
    if len(tr) >= atr_period:
        atr_1d[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr_1d[i] = (tr[i] + (atr_period - 1) * atr_1d[i-1]) / atr_period
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    bb_ma = np.full(len(close_1d), np.nan)
    bb_stddev = np.full(len(close_1d), np.nan)
    for i in range(bb_period, len(close_1d)):
        bb_ma[i] = np.mean(close_1d[i-bb_period:i])
        bb_stddev[i] = np.std(close_1d[i-bb_period:i])
    
    bb_upper = bb_ma + bb_std * bb_stddev
    bb_lower = bb_ma - bb_std * bb_stddev
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_period = 50
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_period + 1)) + 
                         ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volatility filter: current ATR > 1.5x 10-period average
    atr_ma = np.full(n, np.nan)
    atr_ma_period = 10
    for i in range(atr_ma_period, n):
        if not np.isnan(atr_1d_aligned[i-atr_ma_period:i]).all():
            atr_ma[i] = np.nanmean(atr_1d_aligned[i-atr_ma_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, BB, EMA
    start_idx = max(atr_period, bb_period, ema_period, atr_ma_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        atr_ma_val = atr_ma[i]
        
        if position == 0:
            # Long: Price touches lower Bollinger Band + volatility expansion + above 4h EMA50
            if (price <= bb_lower_aligned[i] and 
                atr > 1.5 * atr_ma_val and 
                price > ema_4h_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price touches upper Bollinger Band + volatility expansion + below 4h EMA50
            elif (price >= bb_upper_aligned[i] and 
                  atr > 1.5 * atr_ma_val and 
                  price < ema_4h_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses above Bollinger middle OR loses volatility
            bb_middle = (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2
            if (price >= bb_middle or 
                atr < 0.8 * atr_ma_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses below Bollinger middle OR loses volatility
            bb_middle = (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2
            if (price <= bb_middle or 
                atr < 0.8 * atr_ma_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Band_Touch_Volatility_Expansion_4hEMA50"
timeframe = "4h"
leverage = 1.0