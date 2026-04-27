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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Exponential Moving Average (34-period) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * multiplier) + (ema_34_1d[i-1] * (1 - multiplier))
    
    # Calculate 1-day Average True Range (14-period) for volatility
    tr1 = np.zeros(len(df_1d))
    tr1[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    for i in range(1, len(df_1d)):
        tr1[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    atr_14_1d = np.full(len(tr1), np.nan)
    if len(tr1) >= 14:
        atr_14_1d[13] = np.mean(tr1[:14])
        for i in range(14, len(tr1)):
            atr_14_1d[i] = (tr1[i] * 13/14) + (atr_14_1d[i-1] * 1/14)
    
    # Calculate 1-day Bollinger Bands (20, 2.0)
    sma_20_1d = np.full(len(close_1d), np.nan)
    std_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
            std_20_1d[i] = np.std(close_1d[i-19:i+1])
    
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    
    # Align 1d indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # ATR-based dynamic thresholds
        upper_threshold = ema_34_1d_aligned[i] + (0.5 * atr_14_1d_aligned[i])
        lower_threshold = ema_34_1d_aligned[i] - (0.5 * atr_14_1d_aligned[i])
        
        if position == 0:
            # Long: Price above EMA34 and breaks above upper threshold with volume
            if price > ema_34_1d_aligned[i] and price > upper_threshold and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA34 and breaks below lower threshold with volume
            elif price < ema_34_1d_aligned[i] and price < lower_threshold and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA34 or volatility spike (potential reversal)
            if price < ema_34_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA34 or volatility spike (potential reversal)
            if price > ema_34_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA34_ATR_Volume"
timeframe = "12h"
leverage = 1.0